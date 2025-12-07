import os
import random
import re
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from torch.utils.data import Dataset

# Assuming common.models.PhotoMeta is available or defining a local simplified version
# Adjust import path if common is not directly accessible from finetuning
try:
    from common.models import PhotoMeta
except ImportError:
    # Define a simplified PhotoMeta for finetuning context if not in app path
    from dataclasses import dataclass
    @dataclass
    class PhotoMeta:
        path: str
        original_name: str
        lat: Optional[float] = None
        lon: Optional[float] = None
        alt: Optional[float] = None
        timestamp: Optional[float] = None  # Unix timestamp
        focal_35mm: Optional[float] = None
        orientation: Optional[int] = None
        digital_zoom: Optional[float] = None
        scene_capture_type: Optional[int] = None
        white_balance: Optional[int] = None
        exposure_mode: Optional[int] = None
        flash: Optional[int] = None
        gps_img_direction: Optional[float] = None  # 방위각(도 단위)
        id: str = ""
        job_id: str = ""
        cluster_id: Optional[str] = None
        order_index: Optional[int] = None

class TripletConstructionDataset(Dataset):
    """
    폴더 구조 기반의 Triplet Dataset 생성기
    구조:
        root_dir/
            cluster_01/
                img1.jpg
                img2.jpg
            cluster_02/
                img3.jpg
                ...
    
    역할:
        - Anchor: 특정 클러스터의 이미지
        - Positive: 같은 클러스터의 다른 이미지 (Before-During-After 관계 학습)
        - Negative: 다른 클러스터의 이미지 (비슷하지만 다른 장소 구별 학습)
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort()
        
        # {class_name: [img_path1, img_path2, ...]}
        self.data = {} 
        self.all_images = [] # (img_path, class_idx)

        for idx, cls_name in enumerate(self.classes):
            cls_folder = os.path.join(root_dir, cls_name)
            img_paths = glob(os.path.join(cls_folder, "*.*"))
            # 유효한 이미지 확장자만 필터링
            img_paths = [p for p in img_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if len(img_paths) < 2:
                continue  # Triplet을 만들려면 최소 2장이 필요
                
            self.data[idx] = img_paths
            for img_path in img_paths:
                self.all_images.append((img_path, idx))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        """
        Returns:
            anchor (Tensor), positive (Tensor), negative (Tensor)
        """
        anchor_path, anchor_label = self.all_images[index]

        # 1. Positive: 같은 폴더(클러스터) 내에서 Anchor가 아닌 다른 이미지 선택
        # (공사 전/중/후 사진이 같은 폴더에 있다고 가정)
        pos_paths = self.data[anchor_label]
        pos_path = anchor_path
        if len(pos_paths) > 1:
            while pos_path == anchor_path:
                pos_path = random.choice(pos_paths)
        
        # 2. Negative: 다른 폴더(클러스터)에서 랜덤 선택
        # TODO: 나중에는 Hard Negative Mining을 적용하여 'GPS가 가까운 다른 클러스터'를 뽑는 로직 추가 권장
        neg_label = anchor_label
        while neg_label == anchor_label:
            neg_label = random.choice(list(self.data.keys()))
        
        neg_path = random.choice(self.data[neg_label])

        anchor_img = Image.open(anchor_path).convert('RGB')
        pos_img = Image.open(pos_path).convert('RGB')
        neg_img = Image.open(neg_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img

def load_ground_truth_from_sql_dump(
    filepath: Path, media_root: Path
) -> Dict[str, Dict[str, List[PhotoMeta]]]:
    """
    Parses a PostgreSQL photos table SQL dump to extract ground truth clusters.

    Args:
        filepath: Path to the SQL dump file (containing COPY FROM stdin data).
        media_root: The base directory for photo storage (e.g., Path("backend/src/assets")).

    Returns:
        A dictionary mapping job_id -> cluster_id -> List[PhotoMeta].
        Photos with NULL cluster_id are ignored.
    """
    ground_truth: Dict[str, Dict[str, List[PhotoMeta]]] = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        in_copy_block = False
        for line in f:
            line = line.strip()
            if line.startswith("COPY public.photos"):
                in_copy_block = True
                continue
            if line == r"\.":
                in_copy_block = False
                continue

            if in_copy_block:
                # Example line: pho_lsr3K5QXOP	job_h2PJBZantA	cls_u3ZwYhQapc	3-label-0.jpeg	job_h2PJBZantA/set/3-label-0.jpeg	job_h2PJBZantA/set/3-label-0.jpeg	0	\N	\N	\N	2025-12-03 07:38:35.804331+00	\N
                parts = line.split('\t')
                
                # Ensure enough parts for relevant fields
                if len(parts) < 12: 
                    continue

                photo_id = parts[0]
                job_id = parts[1]
                cluster_id = parts[2] if parts[2] != r'\N' else None
                original_filename = parts[3]
                storage_path_relative = parts[4]
                
                # Ignore photos without a cluster_id (noise/unassigned)
                if cluster_id is None:
                    continue
                
                # Reconstruct absolute path
                full_image_path = str(media_root / storage_path_relative)
                
                # Extract metadata
                meta_lat = float(parts[7]) if parts[7] != r'\N' else None
                meta_lon = float(parts[8]) if parts[8] != r'\N' else None
                
                # meta_timestamp is "timestamp without time zone" in dump, convert to float (unix epoch)
                meta_timestamp_str = parts[9] if parts[9] != r'\N' else None
                meta_timestamp = None
                if meta_timestamp_str:
                    try:
                        # Assuming format like "YYYY-MM-DD HH:MM:SS.microseconds"
                        # Handle potential timezone offset by ignoring or assuming UTC
                        dt_obj = datetime.strptime(meta_timestamp_str.split('.')[0], "%Y-%m-%d %H:%M:%S")
                        meta_timestamp = dt_obj.timestamp()
                    except ValueError:
                        pass # Ignore malformed timestamps
                
                order_index = int(parts[6]) if parts[6] != r'\N' else None

                photo_meta = PhotoMeta(
                    id=photo_id,
                    job_id=job_id,
                    cluster_id=cluster_id,
                    original_name=original_filename,
                    path=full_image_path,
                    lat=meta_lat,
                    lon=meta_lon,
                    timestamp=meta_timestamp,
                    order_index=order_index,
                    # Fill other PhotoMeta fields with defaults or None if not available in dump
                    alt=None, focal_35mm=None, orientation=None, digital_zoom=None,
                    scene_capture_type=None, white_balance=None, exposure_mode=None,
                    flash=None, gps_img_direction=None
                )

                if job_id not in ground_truth:
                    ground_truth[job_id] = {}
                if cluster_id not in ground_truth[job_id]:
                    ground_truth[job_id][cluster_id] = []
                
                ground_truth[job_id][cluster_id].append(photo_meta)
    
    return ground_truth

def split_train_val(
    ground_truth: Dict[str, Dict[str, List[PhotoMeta]]], 
    val_ratio: float = 0.2,
    seed: int = 42
):
    """
    Splits the ground truth data into train and validation sets by job_id.
    
    Args:
        ground_truth: Dictionary mapping job_id -> cluster_id -> List[PhotoMeta]
        val_ratio: Fraction of jobs to use for validation (default 0.2)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple (train_data, val_data)
    """
    job_ids = list(ground_truth.keys())
    random.seed(seed)
    random.shuffle(job_ids)
    
    split_idx = int(len(job_ids) * (1 - val_ratio))
    train_jobs = job_ids[:split_idx]
    val_jobs = job_ids[split_idx:]
    
    train_data = {job_id: ground_truth[job_id] for job_id in train_jobs}
    val_data = {job_id: ground_truth[job_id] for job_id in val_jobs}
    
    return train_data, val_data
