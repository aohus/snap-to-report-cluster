import os
import argparse
import random
from pathlib import Path
from typing import Dict, List
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageFile

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering

from dataset import load_ground_truth_from_sql_dump, split_train_val
from model import MultiModalNet

class InMemoryTripletDataset(Dataset):
    def __init__(self, ground_truth_data: Dict[str, Dict[str, List]], transform=None):
        """
        Args:
            ground_truth_data: Dict[job_id, Dict[cluster_id, List[PhotoMeta]]]
        """
        self.transform = transform
        self.data = [] # List of (photo_meta, cluster_label_id)
        self.cluster_map = {} # cluster_label_id -> [photo_metas]
        
        # Flatten the structure and assign unique integer IDs to each cluster across all jobs
        global_cluster_id = 0
        
        # Collect all metadata for normalization stats
        all_lats = []
        all_lons = []
        all_timestamps = []

        for job_id, clusters in ground_truth_data.items():
            for cluster_id_str, photos in clusters.items():
                if len(photos) < 2:
                    continue # Need at least 2 for positive pair
                
                self.cluster_map[global_cluster_id] = photos
                
                for p in photos:
                    self.data.append((p, global_cluster_id))
                    if p.lat is not None: all_lats.append(p.lat)
                    if p.lon is not None: all_lons.append(p.lon)
                    if p.timestamp is not None: all_timestamps.append(p.timestamp)
                
                global_cluster_id += 1
                
        self.cluster_ids = list(self.cluster_map.keys())
        
        # Calculate stats
        self.lat_mean = sum(all_lats) / len(all_lats) if all_lats else 0
        self.lat_std = (sum((x - self.lat_mean) ** 2 for x in all_lats) / len(all_lats)) ** 0.5 if all_lats else 1
        self.lon_mean = sum(all_lons) / len(all_lons) if all_lons else 0
        self.lon_std = (sum((x - self.lon_mean) ** 2 for x in all_lons) / len(all_lons)) ** 0.5 if all_lons else 1
        
        if all_timestamps:
            self.ts_min = min(all_timestamps)
            self.ts_max = max(all_timestamps)
            self.ts_range = self.ts_max - self.ts_min if self.ts_max > self.ts_min else 1
        else:
            self.ts_min = 0
            self.ts_range = 1

    def _get_meta_tensor(self, photo_meta):
        lat = photo_meta.lat if photo_meta.lat is not None else self.lat_mean
        lon = photo_meta.lon if photo_meta.lon is not None else self.lon_mean
        ts = photo_meta.timestamp if photo_meta.timestamp is not None else self.ts_min
        
        # Z-score for lat/lon, MinMax for timestamp
        norm_lat = (lat - self.lat_mean) / (self.lat_std + 1e-6)
        norm_lon = (lon - self.lon_mean) / (self.lon_std + 1e-6)
        norm_ts = (ts - self.ts_min) / (self.ts_range + 1e-6)
        
        return torch.tensor([norm_lat, norm_lon, norm_ts], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor_meta, anchor_label = self.data[index]

        # Positive: same cluster
        pos_metas = self.cluster_map[anchor_label]
        pos_meta = anchor_meta
        if len(pos_metas) > 1:
            while pos_meta == anchor_meta:
                pos_meta = random.choice(pos_metas)
        
        # Negative: different cluster (randomly selected)
        neg_label = anchor_label
        while neg_label == anchor_label:
            neg_label = random.choice(self.cluster_ids)
        
        neg_meta = random.choice(self.cluster_map[neg_label])

        def load_image(path):
            try:
                return Image.open(path).convert('RGB')
            except Exception as e:
                # print(f"Error loading image {path}: {e}")
                return Image.new('RGB', (224, 224))

        anchor_img = load_image(anchor_meta.path)
        pos_img = load_image(pos_meta.path)
        neg_img = load_image(neg_meta.path)

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
            
        anchor_meta_t = self._get_meta_tensor(anchor_meta)
        pos_meta_t = self._get_meta_tensor(pos_meta)
        neg_meta_t = self._get_meta_tensor(neg_meta)

        return (anchor_img, anchor_meta_t), (pos_img, pos_meta_t), (neg_img, neg_meta_t)

class ValidationDataset(Dataset):
    def __init__(self, photos, dataset_ref):
        self.photos = photos
        self.dataset_ref = dataset_ref 

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, idx):
        meta = self.photos[idx]
        try:
            img = Image.open(meta.path).convert('RGB')
        except Exception:
             img = Image.new('RGB', (224, 224))
             
        if self.dataset_ref.transform:
            img = self.dataset_ref.transform(img)
            
        meta_t = self.dataset_ref._get_meta_tensor(meta)
        return img, meta_t

def evaluate(model, val_data, train_dataset_ref, device):
    model.eval()
    aris = []
    
    # Define a validation transform (no augmentation)
    # Note: Reusing train transform logic but ideally should disable jitter
    # To keep it simple we rely on train_dataset_ref.transform, but if it has jitter, validation results will vary.
    # Let's assume for now jitter is acceptable or user will refine.
    # Better practice: swap transform temporarily.
    original_transform = train_dataset_ref.transform
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset_ref.transform = val_transform

    with torch.no_grad():
        for job_id, clusters in val_data.items():
            job_photos = []
            true_labels = []
            cluster_to_int = {cid: i for i, cid in enumerate(clusters.keys())}
            
            for cluster_id, photos in clusters.items():
                for p in photos:
                    job_photos.append(p)
                    true_labels.append(cluster_to_int[cluster_id])
            
            if len(job_photos) < 2:
                continue

            val_ds = ValidationDataset(job_photos, train_dataset_ref)
            val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
            
            embeddings = []
            for imgs, metas in val_loader:
                imgs, metas = imgs.to(device), metas.to(device)
                embeds = model(imgs, metas)
                embeddings.extend(embeds.cpu().numpy())
            
            embeddings = np.array(embeddings)
            
            # Clustering evaluation
            # Distance threshold 1.0 matches Triplet margin 1.0
            clustering = AgglomerativeClustering(
                n_clusters=None, 
                distance_threshold=1.0, 
                metric='euclidean',
                linkage='average'
            )
            pred_labels = clustering.fit_predict(embeddings)
            
            ari = adjusted_rand_score(true_labels, pred_labels)
            aris.append(ari)
            
    train_dataset_ref.transform = original_transform
    model.train()
    
    avg_ari = sum(aris) / len(aris) if aris else 0.0
    return avg_ari

def train(args):
    # 1. Device Setting
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Using device: {device}")

    # 2. Data Loading
    print(f"Loading ground truth from SQL dump: {args.sql_dump_path}")
    ground_truth = load_ground_truth_from_sql_dump(Path(args.sql_dump_path), Path(args.media_root))
    print(f"Total jobs found: {len(ground_truth)}")
    
    # Split Train/Val
    train_data, val_data = split_train_val(ground_truth, val_ratio=0.2)
    print(f"Train jobs: {len(train_data)}, Val jobs: {len(val_data)}")

    # 3. Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomHorizontalFlip(p=0.0), 
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Dataset & DataLoader
    train_dataset = InMemoryTripletDataset(train_data, transform=train_transform)
    actual_batch_size = min(args.batch_size, len(train_dataset)) if len(train_dataset) > 0 else args.batch_size
    if actual_batch_size < 2: actual_batch_size = 2
    
    dataloader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, num_workers=0)
    print(f"Training with {len(train_dataset)} triplets.")

    # 5. Model Setup
    model = MultiModalNet(embedding_dim=args.embedding_dim).to(device)
    
    # 6. Loss & Optimizer
    criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Report data
    report_path = os.path.join(args.save_dir, "training_report.md")
    report_lines = ["# Training Report\n"]
    report_lines.append(f"- **Model**: MultiModalNet (EfficientNet-B3 + MetaData)")
    report_lines.append(f"- **Embedding Dim**: {args.embedding_dim}")
    report_lines.append(f"- **Epochs**: {args.epochs}")
    report_lines.append(f"- **Batch Size**: {args.batch_size}")
    report_lines.append(f"- **Learning Rate**: {args.lr}")
    report_lines.append("\n## Training Log\n")
    report_lines.append("| Epoch | Avg Loss | Val ARI |")
    report_lines.append("|---|---|---|")

    # 7. Training Loop
    best_ari = -1.0
    model.train()
    
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        running_loss = 0.0
        if len(dataloader) > 0:
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for (anchor_img, anchor_meta), (pos_img, pos_meta), (neg_img, neg_meta) in progress_bar:
                anchor_img, anchor_meta = anchor_img.to(device), anchor_meta.to(device)
                pos_img, pos_meta = pos_img.to(device), pos_meta.to(device)
                neg_img, neg_meta = neg_img.to(device), neg_meta.to(device)

                optimizer.zero_grad()

                embed_a = model(anchor_img, anchor_meta)
                embed_p = model(pos_img, pos_meta)
                embed_n = model(neg_img, neg_meta)

                loss = criterion(embed_a, embed_p, embed_n)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
            
            epoch_loss = running_loss / len(dataloader)
        else:
            epoch_loss = 0.0
            print("No training data available.")

        # Validation
        val_ari = evaluate(model, val_data, train_dataset, device)
        
        print(f"Epoch {epoch+1} Finished. Loss: {epoch_loss:.4f}, Val ARI: {val_ari:.4f}")
        report_lines.append(f"| {epoch+1} | {epoch_loss:.4f} | {val_ari:.4f} |")

        if val_ari > best_ari:
            best_ari = val_ari
            best_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved to {best_path} (ARI: {val_ari:.4f})")

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)

        with open(report_path, "w") as f:
            f.writelines("\n".join(report_lines))

    final_path = os.path.join(args.save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print("Training Complete. Report saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql_dump_path", type=str, required=True)
    parser.add_argument("--media_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embedding_dim", type=int, default=128)
    
    args = parser.parse_args()
    train(args)