import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Import necessary components from dataset.py and existing modules
# Adjust paths if necessary to import from parent directories
sys.path.append(str(Path(__file__).resolve().parent)) # cluster-backend/finetuning
sys.path.append(str(Path(__file__).resolve().parent.parent)) # cluster-backend

from dataset import load_ground_truth_from_sql_dump
from train import EmbeddingNet # Re-use the model definition from train.py

# Import clustering logic from the main application to reuse
# Note: We need to mock or adapt this if it has heavy dependencies
# For this experiment, we'll implement a simple evaluator that uses the trained model embeddings
# and a standard clustering algorithm (e.g., K-Means or hierarchical) to see if it matches ground truth.
# Or better, we can use the Silhouette Score or NMI on the test set.

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def load_model(model_path, device, embedding_dim=128):
    model = EmbeddingNet(embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_features(model, image_paths, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features = []
    valid_paths = []
    
    print("Extracting features...")
    with torch.no_grad():
        for path in tqdm(image_paths):
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                feat = model(img_tensor).cpu().numpy().flatten()
                features.append(feat)
                valid_paths.append(path)
            except Exception as e:
                print(f"Failed to process {path}: {e}")
                
    return np.array(features), valid_paths

def evaluate(ground_truth, model_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    
    print(f"Loading model from {model_path} on {device}...")
    model = load_model(model_path, device)
    
    # Prepare data for evaluation
    # We evaluate per job, as clustering happens per job
    nmi_scores = []
    ari_scores = []
    
    for job_id, clusters in ground_truth.items():
        print(f"Evaluating Job: {job_id}")
        
        job_image_paths = []
        job_labels = []
        
        # Create labels for ground truth
        # Cluster IDs are strings, map them to integers
        cluster_to_int = {cid: i for i, cid in enumerate(clusters.keys())}
        
        for cluster_id, photos in clusters.items():
            if len(photos) == 0: continue
            for p in photos:
                job_image_paths.append(p.path)
                job_labels.append(cluster_to_int[cluster_id])
        
        if len(job_image_paths) < 2:
            print(f"Skipping job {job_id}: Not enough photos.")
            continue
            
        # Extract features
        features, valid_paths = extract_features(model, job_image_paths, device)
        
        # Filter labels for valid paths (in case some images failed to load)
        valid_labels = []
        path_to_label = {p: l for p, l in zip(job_image_paths, job_labels)}
        for p in valid_paths:
            valid_labels.append(path_to_label[p])
            
        if len(valid_labels) < 2:
            continue

        # Perform Clustering (K-Means with known K for evaluation metric)
        # In a real scenario, we don't know K, but here we want to see 
        # if the embeddings *can* separate the classes correctly.
        # So we use K-Means with true number of clusters to measure embedding quality.
        from sklearn.cluster import KMeans
        n_clusters = len(set(valid_labels))
        if n_clusters < 2:
             print(f"Skipping job {job_id}: Only 1 cluster found.")
             continue
             
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        predicted_labels = kmeans.fit_predict(features)
        
        # Calculate Metrics
        nmi = normalized_mutual_info_score(valid_labels, predicted_labels)
        ari = adjusted_rand_score(valid_labels, predicted_labels)
        
        nmi_scores.append(nmi)
        ari_scores.append(ari)
        print(f"Job {job_id} -> NMI: {nmi:.4f}, ARI: {ari:.4f}")

    print("\n=== Evaluation Results ===")
    print(f"Average NMI: {np.mean(nmi_scores):.4f}")
    print(f"Average ARI: {np.mean(ari_scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Finetuned Model")
    parser.add_argument("--sql_dump_path", type=str, required=True, help="Path to the .sql file")
    parser.add_argument("--media_root", type=str, required=True, help="Path to the media root directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model .pth file")

    args = parser.parse_args()
    
    gt = load_ground_truth_from_sql_dump(Path(args.sql_dump_path), Path(args.media_root))
    evaluate(gt, args.model_path)
