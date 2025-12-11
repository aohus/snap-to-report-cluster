# flake8: noqa
from __future__ import annotations

import asyncio
import csv
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# OpenCV
import cv2
import kornia
import kornia.feature as KF

# matplotlib
import matplotlib.pyplot as plt
import numpy as np

# CLIP
import open_clip
import timm

# Torch / vision
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# sklearn
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain.extractors.metadata_extractor import MetadataExtractor
from photometa import PhotoMeta

# ===================================================================
# Global Setup
# ===================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
logger.info(f"Using device: {DEVICE}")


# ===================================================================
# NEW: Semantic Segmenter
# ===================================================================
class SemanticSegmenter:
    """
    Detects specific classes in an image using a deep learning model
    and generates a mask. The generated masks are cached for reuse.
    """
    def __init__(self, model_name: str, classes_to_mask: List[str], device: torch.device):
        self.device = device
        self.model = torch.hub.load('pytorch/vision:v0.13.0', model_name, pretrained=True)
        self.model.to(self.device).eval()
        
        self.coco_class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        self.mask_class_indices = {self.coco_class_names.index(name) for name in classes_to_mask if name in self.coco_class_names}
        logger.info(f"[SemanticSegmenter] Masking classes: {[self.coco_class_names[i] for i in self.mask_class_indices]}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._mask_cache: Dict[Path, np.ndarray] = {}

    @torch.no_grad()
    def create_mask(self, image_path: Path) -> np.ndarray:
        if image_path in self._mask_cache:
            return self._mask_cache[image_path]

        try:
            input_image = Image.open(image_path).convert("RGB")
            original_size = input_image.size
            input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)

            output = self.model(input_tensor)['out'][0]
            output_predictions = output.argmax(0).byte().cpu()
            
            output_pil = to_pil_image(output_predictions).resize(original_size, Image.NEAREST)
            output_predictions = to_tensor(output_pil).squeeze(0).byte()

            mask = torch.ones_like(output_predictions, dtype=torch.uint8)
            for class_idx in self.mask_class_indices:
                mask[output_predictions == class_idx] = 0

            mask_np = mask.numpy()
            self._mask_cache[image_path] = mask_np
            return mask_np
        except Exception as e:
            logger.warning(f"Failed to create semantic mask for {image_path}: {e}")
            img = Image.open(image_path)
            return np.ones((img.height, img.width), dtype=np.uint8)


# ===================================================================
# Core Clustering Components (Adapted from original script)
# ===================================================================

# --- DataClasses ---
@dataclass
class GPSParams:
    eps_m: float
    min_samples: int

@dataclass
class EmbedParams:
    name: str
    w_apgem: float
    w_clip: float
    similarity_threshold: float
    knn_k: int

@dataclass
class GeoParams:
    matcher_type: str
    geo_threshold: float
    # SIFT
    max_features: Optional[int] = None
    ratio_thresh: Optional[float] = None
    ransac_reproj_thresh: Optional[float] = None
    min_good_matches: Optional[int] = None
    # LoFTR
    confidence_threshold: Optional[float] = None
    # Semantic Masking
    use_semantic_mask: bool = False
    semantic_model_name: str = 'deeplabv3_resnet101'
    semantic_classes_to_mask: List[str] = field(default_factory=lambda: [
        'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'boat',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'potted plant', 'tv',
        'laptop', 'chair', 'couch', 'dining table'
    ])

@dataclass
class ExperimentConfig:
    id: str
    gps: GPSParams
    embed: EmbedParams
    geo: GeoParams

# --- GPS Clustering ---
def latlon_to_xy_m(lat, lon, lat0, lon0):
    lat_rad, lon_rad = math.radians(lat), math.radians(lon)
    lat0_rad, lon0_rad = math.radians(lat0), math.radians(lon0)
    x = (lon_rad - lon0_rad) * math.cos((lat_rad + lat0_rad) / 2.0) * 6371000.0
    y = (lat_rad - lat0_rad) * 6371000.0
    return x, y

def gps_dbscan_cluster(photos, eps_m, min_samples):
    if not photos: return []
    lat0 = sum(p.lat for p in photos) / len(photos)
    lon0 = sum(p.lon for p in photos) / len(photos)
    coords = np.array([latlon_to_xy_m(p.lat, p.lon, lat0, lon0) for p in photos], dtype=np.float32)
    return DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(coords).tolist()

# --- Descriptor Extractors ---
class BaseDescriptorExtractor:
    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        raise NotImplementedError

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class APGeMDescriptorExtractor(BaseDescriptorExtractor):
    def __init__(self, model_name="tf_efficientnet_b3_ns", image_size=320, device=DEVICE):
        self.device = device
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="").to(device).eval()
        self.pool = GeM().to(device)
        self.transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
    @torch.no_grad()
    def extract_one(self, image_path: Path):
        try:
            img = Image.open(image_path).convert("RGB")
            x = self.transform(img).unsqueeze(0).to(self.device)
            feat_map = self.backbone(x)
            desc = self.pool(feat_map).flatten(1)
            return F.normalize(desc, p=2, dim=1).squeeze(0).cpu().numpy()
        except Exception as e:
            logger.warning(f"[APGeM] Failed on {image_path}: {e}")
            return None

class CLIPDescriptorExtractor(BaseDescriptorExtractor):
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device=DEVICE):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(device).eval()
    @torch.no_grad()
    def extract_one(self, image_path: Path):
        try:
            img = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(img).unsqueeze(0).to(self.device)
            features = self.model.encode_image(image_input)
            return F.normalize(features, p=2, dim=-1).cpu().numpy().flatten()
        except Exception as e:
            logger.warning(f"[CLIP] Failed on {image_path}: {e}")
            return None

class CombinedAPGeMCLIPExtractor(BaseDescriptorExtractor):
    def __init__(self, w_apgem=0.7, w_clip=0.3):
        self.apgem = APGeMDescriptorExtractor()
        self.clip = CLIPDescriptorExtractor()
        self.w_apgem, self.w_clip = w_apgem, w_clip
    def extract_one(self, image_path: Path):
        f_ap = self.apgem.extract_one(image_path)
        f_cl = self.clip.extract_one(image_path)
        if f_ap is None or f_cl is None: return None
        
        f_ap = f_ap / np.linalg.norm(f_ap) * self.w_apgem
        f_cl = f_cl / np.linalg.norm(f_cl) * self.w_clip
        combined = np.concatenate([f_ap, f_cl])
        return combined / np.linalg.norm(combined)

# --- Geometry Matchers ---
class LocalGeometryMatcher: # SIFT
    def __init__(self, max_features=1500, ratio_thresh=0.75, ransac_reproj_thresh=5.0, min_good_matches=10):
        try:
            self.detector = cv2.SIFT_create(nfeatures=max_features)
            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            self.enabled = True
        except Exception:
            logger.warning("SIFT not available (opencv-contrib-python needed). SIFT matcher disabled.")
            self.enabled = False
        self.ratio_thresh = ratio_thresh
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.min_good_matches = min_good_matches

    def geo_score(self, path1, path2):
        if not self.enabled: return 0.0
        img1 = cv2.imread(str(path1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(path2), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None: return 0.0

        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2: return 0.0
        
        matches = self.bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < self.ratio_thresh * n.distance]
        
        if len(good) < self.min_good_matches: return 0.0
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_reproj_thresh)
        return float(np.sum(mask)) / len(good) if H is not None else 0.0

class LoFTRMatcher:
    def __init__(self, confidence_threshold=0.8, pretrained="outdoor", device=DEVICE, segmenter=None):
        self.confidence_threshold = confidence_threshold
        self.loftr = KF.LoFTR(pretrained=pretrained).to(device)
        self.device, self.segmenter = device, segmenter

    def _load_gray_tensor(self, path):
        try:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (640, 480))
            tensor = kornia.image_to_tensor(img, keepdim=False).float() / 255.0

            # Ensure tensor is 4D for LoFTR
            if tensor.dim() == 3: # Should be (1,H,W)
                tensor = tensor.unsqueeze(0) # Becomes (1,1,H,W)
            
            if self.segmenter:
                mask = self.segmenter.create_mask(path)
                mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)
                bool_mask = torch.from_numpy(mask) == 0
                tensor.squeeze()[bool_mask] = 0.0
            return tensor.to(self.device)
        except Exception as e:
            logger.warning(f"Failed to load/process {path} for LoFTR: {e}")
            return None

    @torch.no_grad()
    def geo_score(self, path1, path2):
        t1, t2 = self._load_gray_tensor(path1), self._load_gray_tensor(path2)
        if t1 is None or t2 is None: return 0.0
        correspondences = self.loftr({"image0": t1, "image1": t2})
        num_matches = np.sum(correspondences['confidence'].cpu().numpy() > self.confidence_threshold)
        return min(num_matches / 100.0, 1.0) # Heuristic normalization

# --- Clustering Logic ---
class DeepCluster:
    def __init__(self, descriptor_extractor, geo_matcher, similarity_threshold, geo_threshold, knn_k):
        self.descriptor = descriptor_extractor
        self.geo_matcher = geo_matcher
        self.sim_th, self.geo_th, self.knn_k = similarity_threshold, geo_threshold, knn_k

    def cluster(self, photos):
        if len(photos) < 2: return [[p] for p in photos]
        
        feats, valid_photos = self._extract_features(photos)
        if len(valid_photos) < 2: return [[p] for p in valid_photos]

        edges = self._build_candidate_edges(feats, valid_photos)
        filtered_edges = self._filter_by_geometry(edges, valid_photos)
        labels = self._connected_components(len(valid_photos), filtered_edges)
        return self._build_clusters(labels, valid_photos)

    def _extract_features(self, photos):
        feats, valid = [], []
        for p in photos:
            v = self.descriptor.extract_one(p.path)
            if v is not None:
                feats.append(v)
                valid.append(p)
        return (np.stack(feats), valid) if feats else (np.empty((0,0)), [])

    def _build_candidate_edges(self, features, photos):
        n = features.shape[0]
        k = min(self.knn_k, n)
        nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(features)
        distances, indices = nn.kneighbors(features)
        
        edges = set()
        for i in range(n):
            for dist, j in zip(distances[i][1:], indices[i][1:]):
                if (1.0 - dist) >= self.sim_th:
                    edges.add(tuple(sorted((i, j))))
        return list(edges)

    def _filter_by_geometry(self, edges, photos):
        if self.geo_matcher is None or not edges: return edges
        return [ (i, j) for i, j in edges if self.geo_matcher.geo_score(photos[i].path, photos[j].path) >= self.geo_th ]

    def _connected_components(self, n, edges):
        parent = list(range(n))
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
        def union(i, j):
            root_i, root_j = find(i), find(j)
            if root_i != root_j: parent[root_j] = root_i
        for i, j in edges: union(i, j)
        return [find(i) for i in range(n)]

    def _build_clusters(self, labels, photos):
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(photos[i])
        return sorted([sorted(c, key=lambda p:p.timestamp) for c in clusters.values()], key=len, reverse=True)


# ===================================================================
# Experiment Orchestration
# ===================================================================
def create_deep_cluster(embed_params, geo_params, segmenter):
    descriptor = CombinedAPGeMCLIPExtractor(w_apgem=embed_params.w_apgem, w_clip=embed_params.w_clip)
    geo_matcher = None
    if geo_params.matcher_type == "sift":
        geo_matcher = LocalGeometryMatcher(
            max_features=geo_params.max_features, ratio_thresh=geo_params.ratio_thresh,
            ransac_reproj_thresh=geo_params.ransac_reproj_thresh, min_good_matches=geo_params.min_good_matches)
    elif geo_params.matcher_type == "loftr":
        geo_matcher = LoFTRMatcher(
            confidence_threshold=geo_params.confidence_threshold, device=DEVICE,
            segmenter=segmenter if geo_params.use_semantic_mask else None)
    
    return DeepCluster(descriptor, geo_matcher, embed_params.similarity_threshold, geo_params.geo_threshold, embed_params.knn_k)

def write_summary_row(summary_path, exp, ari, nmi, n_clusters, stage_time, **kwargs) -> None:
    data = kwargs
    columns = [
        "exp_id", 
        "embed_name", 
        "sim_th", 
        "knn_k", 
        "geo_matcher", 
        "geo_th", 
        "use_mask", 
        "ARI", 
        "NMI", 
        "n_clusters", 
        "time_sec"]
    rows = [
        exp.id, 
        exp.embed.name, 
        exp.embed.similarity_threshold, 
        exp.embed.knn_k, 
        exp.geo.matcher_type, 
        exp.geo.geo_threshold, 
        exp.geo.use_semantic_mask, 
        f"{ari:.4f}", 
        f"{nmi:.4f}", 
        n_clusters, 
        f"{stage_time:.2f}"
    ]

    if data:
        for k, v in data:
            columns.add(k)
            rows.add(v)
        
    header = not summary_path.exists()
    with summary_path.open("a") as f:
        writer = csv.writer(f)
        if header: writer.writerow(columns)
        writer.writerow(rows)


def run_one_experiment(exp, photos, true_labels, out_dir, summary_path, segmenter):
    logger.info(f"--- Running Experiment: {exp.id} ---")
    exp_dir = out_dir / exp.id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = time.perf_counter()
    gps_labels = gps_dbscan_cluster(photos, exp.gps.eps_m, exp.gps.min_samples)
    
    gps_clusters = defaultdict(list)
    for i, label in enumerate(gps_labels):
        if label != -1:
            gps_clusters[label].append(photos[i])

    final_labels, next_label = [-1] * len(photos), 0
    photo_to_idx = {p.id: i for i, p in enumerate(photos)}
    
    for gps_photos in gps_clusters.values():
        clusterer = create_deep_cluster(exp.embed, exp.geo, segmenter)
        scene_clusters = clusterer.cluster(gps_photos)
        for cluster in scene_clusters:
            for photo in cluster:
                final_labels[photo_to_idx[photo.id]] = next_label
            next_label += 1
            
    stage_time = time.perf_counter() - t0
    n_clusters = len(set(l for l in final_labels if l >= 0))
    ari = adjusted_rand_score(true_labels, final_labels)
    nmi = normalized_mutual_info_score(true_labels, final_labels)
    
    logger.info(f"  -> Results: ARI={ari:.4f}, NMI={nmi:.4f}, Clusters={n_clusters}, Time={stage_time:.2f}s")
    
    # Save artifacts
    create_mosaic_for_experiment(photos, final_labels, exp_dir / f"{exp.id}_mosaic.png")
    write_summary_row(summary_path, exp, ari=ari, nmi=nmi, n_clusters=n_clusters, stage_time=stage_time)

def create_mosaic_for_experiment(photos, labels, out_path, thumb_size=(256, 256), max_per_cluster=10):
    clusters = defaultdict(list)
    for i, l in enumerate(labels):
        if l != -1: clusters[l].append(photos[i])
    if not clusters: return
    
    n_clusters = len(clusters)
    mosaic = Image.new('RGB', (thumb_size[0] * max_per_cluster, thumb_size[1] * n_clusters), (255,255,255))
    
    for i, (label, cluster_photos) in enumerate(sorted(clusters.items())):
        for j, p in enumerate(cluster_photos[:max_per_cluster]):
            try:
                img = Image.open(p.path).resize(thumb_size)
                mosaic.paste(img, (j * thumb_size[0], i * thumb_size[1]))
            except: continue
    out_path.parent.mkdir(exist_ok=True)
    mosaic.save(out_path)

def build_experiment_grid():
    configs = []
    idx = 0
    
    # Define the parameter grid
    sim_thresholds = [0.76, 0.78, 0.80, 0.82, 0.85, 0.89]
    knn_ks = [8, 10]
    geo_thresholds = [0.1, 0.15, 0.2, 0.25]
    
    gps_params = GPSParams(eps_m=18.0, min_samples=3)

    # Grid search over the parameters
    for sim in sim_thresholds:
        for k in knn_ks:
            for geo_th in geo_thresholds:
                embed_params = EmbedParams(name="APGeM+CLIP", w_apgem=0.7, w_clip=0.3, similarity_threshold=sim, knn_k=k)
                
                # Using LoFTR with mask as the base for optimization
                geo_params = GeoParams(
                    matcher_type="loftr", 
                    geo_threshold=geo_th, 
                    confidence_threshold=0.8, # Keep this constant for now
                    use_semantic_mask=True
                )

                exp_id = f"exp_{idx:03d}_sim{sim}_k{k}_geo{geo_th}"
                configs.append(ExperimentConfig(id=exp_id, gps=gps_params, embed=embed_params, geo=geo_params))
                idx += 1
                
    return configs

async def get_photos(image_dir):
    image_paths = [p for p in Path(image_dir).glob('**/*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    meta_ext = MetadataExtractor()
    return await asyncio.gather(*[meta_ext.extract(str(p)) for p in image_paths])

if __name__ == "__main__":
    exp_name = "semantic_2"
    summary_path = Path(f"./.experiment/exp_results/summary/summary_{exp_name}.csv")
    out_dir = Path(f"./.experiment/exp_results/{exp_name}")
    image_dir = Path("./assets/set1/") # Use relative path
    
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir.resolve()}")
        sys.exit(1)

    photos = asyncio.run(get_photos(image_dir))
    # true_labels = [int(p.original_name.split("-")[0]) for p in photos]
    true_labels = None
    
    default_geo_params = GeoParams(matcher_type='loftr', geo_threshold=0.15)
    segmenter = SemanticSegmenter(
        model_name=default_geo_params.semantic_model_name,
        classes_to_mask=default_geo_params.semantic_classes_to_mask,
        device=DEVICE
    )

    for exp_config in build_experiment_grid():
        run_one_experiment(exp_config, photos, true_labels, out_dir, summary_path, segmenter)

    logger.info(f"All experiments finished. Results are in {out_dir.resolve()}")
