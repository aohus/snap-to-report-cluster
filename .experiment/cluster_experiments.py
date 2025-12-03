from __future__ import annotations

import asyncio
import csv
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# OpenCV
import cv2  # type: ignore

# matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open_clip

# timm & CLIP
import timm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
# Torch / vision
import torch
import torch.nn as nn
import torch.nn.functional as F
from domain.extractors.metadata_extractor import MetadataExtractor
from photometa import PhotoMeta
from PIL import Image

# sklearn
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from dataclasses import dataclass
from typing import Optional


@dataclass
class PairDebugRecord:
    photo_path_i: str
    photo_path_j: str
    base_sim: float
    eff_sim: float
    passed_sim: bool
    geo_score: Optional[float]
    passed_geo: Optional[bool]
    gps_cluster_index: int  # 이 쌍이 속한 GPS 클러스터 번호

# -------------------------------------------------------------------
# Stage 1: GPS DBSCAN
# -------------------------------------------------------------------

EARTH_RADIUS_M = 6371000.0  # meters


def latlon_to_xy_m(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)
    x = (lon_rad - lon0_rad) * math.cos((lat_rad + lat0_rad) / 2.0) * EARTH_RADIUS_M
    y = (lat_rad - lat0_rad) * EARTH_RADIUS_M
    return x, y


def gps_dbscan_cluster(
    photos: List[PhotoMeta],
    eps_m: float,
    min_samples: int,
) -> List[int]:
    if not photos:
        return []
    lat0 = sum(p.lat for p in photos) / len(photos)
    lon0 = sum(p.lon for p in photos) / len(photos)

    coords = np.array(
        [latlon_to_xy_m(p.lat, p.lon, lat0, lon0) for p in photos],
        dtype=np.float32,
    )
    db = DBSCAN(eps=eps_m, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(coords)
    return labels.tolist()


# -------------------------------------------------------------------
# 공통 Descriptor 인터페이스
# -------------------------------------------------------------------

class BaseDescriptorExtractor:
    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        raise NotImplementedError


# -------------------------------------------------------------------
# GeM + APGeMDescriptorExtractor
# -------------------------------------------------------------------

class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps)
        x = x ** self.p
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))
        x = x ** (1.0 / self.p)
        return x  # (B, C, 1, 1)


class APGeMDescriptorExtractor(BaseDescriptorExtractor):
    """
    EfficientNet + GeM 기반 AP-GeM 계열 글로벌 디스크립터
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnet_b3_ns",
        image_size: int = 320,
        device: Optional[str] = None,
    ) -> None:
        if device is not None:
            self.device = torch.device(device)
        else:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        logger.info(f"[APGeM] using device = {self.device}")

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        self.backbone.to(self.device)
        self.backbone.eval()

        self.feat_dim = self.backbone.num_features
        logger.info(f"[APGeM] backbone={model_name}, feat_dim={self.feat_dim}")

        self.pool = GeM(p=3.0).to(self.device)

        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN,
                    std=IMAGENET_DEFAULT_STD,
                ),
            ]
        )

    @torch.no_grad()
    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"[APGeM] Failed to open image {image_path}: {e}")
            return None

        x = self.transform(img).unsqueeze(0).to(self.device)

        feat_map = self.backbone(x)

        if feat_map.ndim == 2:
            desc = feat_map
        else:
            pooled = self.pool(feat_map)
            desc = pooled.flatten(1)

        desc = F.normalize(desc, p=2, dim=1)
        return (
            desc.squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )


# -------------------------------------------------------------------
# CLIPDescriptorExtractor
# -------------------------------------------------------------------
class StructureScorer:
    """
    CLIP 텍스트 프롬프트를 이용해서
    - building/structure
    - vegetation(나무/풀)
    사이의 상대적인 점수 차이로 구조물 점수를 계산.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        if device is None:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model.to(self.device).eval()

        texts = [
            "a scene with buildings, walls, stairs, benches, fences, paved roads, trees, bushes, grass, plants",
            "a scene with people, cars, machines",
        ]
        with torch.no_grad():
            toks = open_clip.tokenize(texts).to(self.device)
            txt_feat = self.model.encode_text(toks)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        self.t_building = txt_feat[0]  # 구조물
        self.t_veget = txt_feat[1]     # 식생

    @torch.no_grad()
    def score_from_image(self, img: Image.Image) -> float:
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        img_feat = self.model.encode_image(x)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        sims = (img_feat @ torch.stack([self.t_building, self.t_veget]).T).squeeze(0)
        s_building = sims[0].item()
        s_veget = sims[1].item()

        # building - vegetation 차이를 sigmoid로 0~1로 압축
        alpha = 5.0
        score = 1.0 / (1.0 + math.exp(-alpha * (s_building - s_veget)))
        return float(score)

    @torch.no_grad()
    def score_from_path(self, path: Path) -> float:
        img = Image.open(path).convert("RGB")
        return self.score_from_image(img)
    
class CLIPDescriptorExtractor(BaseDescriptorExtractor):
    """
    OpenCLIP 기반 글로벌 디스크립터
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
    ) -> None:
        if device is not None:
            self.device = torch.device(device)
        else:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        logger.info(f"[CLIP] using device = {self.device}")

        self.model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = preprocess

    @torch.no_grad()
    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"[CLIP] Failed to open image {image_path}: {e}")
            return None

        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        features = self.model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)
        feat_np = features.cpu().numpy().flatten().astype(np.float32)
        return feat_np


# -------------------------------------------------------------------
# Combined APGeM + CLIP Extractor
# -------------------------------------------------------------------

class CombinedAPGeMCLIPExtractor(BaseDescriptorExtractor):
    """
    APGeM + CLIP 조합 디스크립터
    - 각 디스크립터를 L2 normalize한 뒤, 가중치 곱해 concat
    - 마지막에 한 번 더 L2 normalize (선택)
    """

    def __init__(
        self,
        apgem: Optional[APGeMDescriptorExtractor] = None,
        clip: Optional[CLIPDescriptorExtractor] = None,
        w_apgem: float = 0.7,
        w_clip: float = 0.3,
        l2_normalize_final: bool = True,
    ) -> None:
        # 같은 device를 쓰는 게 좋음 (여기서는 각각 내부에서 device 결정)
        self.apgem = apgem or APGeMDescriptorExtractor()
        self.clip = clip or CLIPDescriptorExtractor()
        self.w_apgem = float(w_apgem)
        self.w_clip = float(w_clip)
        self.l2_normalize_final = bool(l2_normalize_final)

    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        f_ap = self.apgem.extract_one(image_path)
        f_cl = self.clip.extract_one(image_path)

        if f_ap is None and f_cl is None:
            return None

        parts: List[np.ndarray] = []

        if f_ap is not None:
            fa = f_ap.astype(np.float32)
            fa = fa / (np.linalg.norm(fa) + 1e-8)
            fa = fa * self.w_apgem
            parts.append(fa)

        if f_cl is not None:
            fc = f_cl.astype(np.float32)
            fc = fc / (np.linalg.norm(fc) + 1e-8)
            fc = fc * self.w_clip
            parts.append(fc)

        if not parts:
            return None

        combined = np.concatenate(parts).astype(np.float32)

        if self.l2_normalize_final:
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm

        return combined


# -------------------------------------------------------------------
# Stage 3: Local Geometry (SIFT + RANSAC)
# -------------------------------------------------------------------

class LocalGeometryMatcher:
    def __init__(
        self,
        max_features: int = 1500,
        ratio_thresh: float = 0.75,
        ransac_reproj_thresh: float = 3.0,
        min_good_matches: int = 15,
    ) -> None:
        self.max_features = max_features
        self.ratio_thresh = ratio_thresh
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.min_good_matches = min_good_matches

        try:
            self.detector = cv2.SIFT_create(nfeatures=self.max_features)  # type: ignore[attr-defined]
            self.enabled = True
        except Exception:
            logger.warning(
                "SIFT 생성 실패 (opencv-contrib-python 필요). "
                "기하 검증 비활성화, 항상 score=1.0 반환."
            )
            self.detector = None
            self.enabled = False

    def _load_gray(self, path: Path) -> Optional[np.ndarray]:
        if not self.enabled or self.detector is None:
            return None
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # type: ignore[attr-defined]
        if img is None:
            logger.warning(f"Failed to read image for geometry: {path}")
            return None
        return img

    def geo_score(self, path1: Path, path2: Path) -> float:
        if not self.enabled or self.detector is None:
            return 1.0

        img1 = self._load_gray(path1)
        img2 = self._load_gray(path2)
        if img1 is None or img2 is None:
            return 0.0

        keypoints1, desc1 = self.detector.detectAndCompute(img1, None)
        keypoints2, desc2 = self.detector.detectAndCompute(img2, None)
        if desc1 is None or desc2 is None:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # type: ignore[attr-defined]
        matches = bf.knnMatch(desc1, desc2, k=2)  # type: ignore[attr-defined]

        good = []
        for m, n in matches:
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)

        if len(good) < self.min_good_matches:
            return 0.0

        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(  # type: ignore[attr-defined]
            pts1,
            pts2,
            cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_thresh,
        )
        if H is None or mask is None:
            return 0.0

        inliers = int(mask.ravel().sum())
        total = len(good)
        if total == 0:
            return 0.0
        score = float(inliers) / float(total)
        return max(0.0, min(1.0, score))


# -------------------------------------------------------------------
# Union-Find
# -------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def to_labels(self) -> List[int]:
        roots = [self.find(i) for i in range(len(self.parent))]
        root_to_label: Dict[int, int] = {}
        labels: List[int] = []
        next_label = 0
        for r in roots:
            if r not in root_to_label:
                root_to_label[r] = next_label
                next_label += 1
            labels.append(root_to_label[r])
        return labels


# -------------------------------------------------------------------
# Stage 2~5: DeepCluster
# -------------------------------------------------------------------

class DeepCluster:
    """
    GPS 클러스터 내에서:
      - 전역 임베딩
      - k-NN 그래프
      - SIFT/RANSAC 기하 검증
      - Union-Find 연결요소
      - 촬영 시각 순 정렬
    """

    def __init__(
        self,
        descriptor_extractor,
        geo_matcher=None,
        similarity_threshold: float = 0.8,
        geo_threshold: float = 0.3,
        knn_k: int = 10,
        min_cluster_size: int = 2,
        structure_scores: Dict[str, float] | None = None,
        enable_debug: bool = False,
    ) -> None:
        self.descriptor_extractor = descriptor_extractor
        self.geo_matcher = geo_matcher
        self.similarity_threshold = similarity_threshold
        self.geo_threshold = geo_threshold
        self.knn_k = knn_k
        self.min_cluster_size = min_cluster_size
        # photo.id -> 0~1
        self.structure_scores = structure_scores or {}
        self._pair_debug: list[dict] = []
        self.enable_debug = enable_debug

    def cluster(self, photos: Sequence[PhotoMeta]) -> List[List[PhotoMeta]]:
        if not photos:
            return []

        logger.info(f"DeepCluster: {len(photos)} photos")

        features, valid_photos = self._extract_features(photos)
        if len(valid_photos) < 2:
            return [[p] for p in valid_photos]

        edges = self._build_candidate_edges(features, photos=photos)
        edges = self._filter_edges_by_geometry(edges, valid_photos)

        labels = self._connected_components(len(valid_photos), edges)
        clusters = self._build_clusters_from_labels(labels, valid_photos)
        return clusters, self._pair_debug

    def _extract_features(
        self, photos: Sequence[PhotoMeta]
    ) -> Tuple[np.ndarray, List[PhotoMeta]]:
        feats: List[np.ndarray] = []
        valid: List[PhotoMeta] = []
        for p in photos:
            v = self.descriptor_extractor.extract_one(p.path)
            if v is None:
                continue
            feats.append(v.astype(np.float32))
            valid.append(p)
        if not feats:
            return np.empty((0, 0), dtype=np.float32), []
        arr = np.stack(feats, axis=0)
        return arr, valid
    
    # def _build_candidate_edges(
    #     self,
    #     features: np.ndarray,
    #     photos: List["PhotoMeta"],
    # ) -> List[Tuple[int, int]]:
    #     n, d = features.shape
    #     if n == 0:
    #         return []
        
    #     self._pair_debug = []  # 클러스터링마다 초기화
    #     k = min(max(2, self.knn_k), n)
    #     logger.info(f"🔗 k-NN graph 구성 (n={n}, d={d}, k={k})")

    #     nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    #     nn.fit(features)
    #     distances, indices = nn.kneighbors(features)

    #     edges: List[Tuple[int, int]] = []

    #     for i in range(n):
    #         # indices[i][0] == 자기 자신
    #         pid_i = photos[i].original_name
    #         for dist, j in zip(distances[i][1:], indices[i][1:]):
    #             pid_j = photos[j].original_name
    #             sim = 1.0 - float(dist)
    #             passed_sim = sim >= self.similarity_threshold
    #             if self.enable_debug:
    #                 rec = {
    #                     "i": i,
    #                     "j": j,
    #                     "photo_path_i": pid_i,
    #                     "photo_path_j": pid_j,
    #                     "base_sim": sim,
    #                     "eff_sim": sim, 
    #                     "passed_sim": passed_sim,
    #                     "geo_score": None,
    #                     "passed_geo": None,
    #                 }
                
    #             if sim < self.similarity_threshold:
    #                 continue
                
    #             # 아직 기하 검증은 하지 않고, 후보로만 저장 (Stage 3에서 필터링)
    #             if i < j:
    #                 edges.append((i, j))
    #                 self._pair_debug.append(rec)
    #     logger.info(f"🔍 전역 임베딩 기준 후보 edge 수: {len(edges)}")
    #     return edges
    
    def _build_candidate_edges(
        self,
        features: np.ndarray,
        photos: list["PhotoMeta"],
    ) -> list[tuple[int, int]]:
        n, d = features.shape
        if n == 0:
            return []

        self._pair_debug = []  # 클러스터링마다 초기화

        k = min(max(2, self.knn_k), n)
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(features)
        distances, indices = nn.kneighbors(features)

        edges: list[tuple[int, int]] = []

        for i in range(n):
            pid_i = photos[i].original_name
            s_i = self.structure_scores.get(pid_i, 0.5)

            for dist, j in zip(distances[i][1:], indices[i][1:]):
                pid_j = photos[j].original_name
                s_j = self.structure_scores.get(pid_j, 0.5)

                base_sim = 1.0 - float(dist)
                struct_w = 0.5 + 0.5 * min(s_i, s_j)  # 0.5~1.0
                eff_sim = base_sim * struct_w

                passed_sim = eff_sim >= self.similarity_threshold

                # 디버그 기록 (Stage 2 정보)
                if self.enable_debug:
                    rec = {
                        "i": i,
                        "j": j,
                        "photo_path_i": pid_i,
                        "photo_path_j": pid_j,
                        "base_sim": base_sim,
                        "eff_sim": eff_sim,
                        "passed_sim": passed_sim,
                        "geo_score": None,
                        "passed_geo": None,
                    }
                    self._pair_debug.append(rec)

                if not passed_sim:
                    continue
                if i < j:
                    edges.append((i, j))
        logger.info(f"🔍 전역 임베딩 기준 후보 edge 수: {len(edges)}")
        return edges

    def _filter_edges_by_geometry(
        self,
        edges: list[tuple[int, int]],
        photos: list["PhotoMeta"],
    ) -> list[tuple[int, int]]:
        if self.geo_matcher is None or not edges:
            return edges

        kept: list[tuple[int, int]] = []

        for i, j in edges:
            p_i = photos[i]
            p_j = photos[j]

            geo_score = self.geo_matcher.geo_score(p_i.path, p_j.path)
            passed_geo = geo_score >= self.geo_threshold

            # 디버그 정보 업데이트
            if self.enable_debug:
                for rec in self._pair_debug:
                    if rec["i"] == i and rec["j"] == j:
                        rec["geo_score"] = float(geo_score)
                        rec["passed_geo"] = bool(passed_geo)
                        break

            if passed_geo:
                kept.append((i, j))

        return kept

    def _connected_components(self, n: int, edges: List[Tuple[int, int]]) -> List[int]:
        if n == 0:
            return []
        if not edges:
            return list(range(n))
        uf = UnionFind(n)
        for i, j in edges:
            uf.union(i, j)
        labels = uf.to_labels()
        return labels

    def _build_clusters_from_labels(
        self,
        labels: List[int],
        photos: List[PhotoMeta],
    ) -> List[List[PhotoMeta]]:
        label_to_items: Dict[int, List[PhotoMeta]] = {}
        for idx, lbl in enumerate(labels):
            label_to_items.setdefault(lbl, []).append(photos[idx])

        clusters: List[List[PhotoMeta]] = []
        for lbl, items in label_to_items.items():
            if len(items) < self.min_cluster_size:
                # 필요하면 싱글톤도 살릴 수 있음
                clusters.append(sorted(items, key=lambda p: (p.timestamp or datetime.min, str(p.id))))
            else:
                clusters.append(sorted(items, key=lambda p: (p.timestamp or datetime.min, str(p.id))))
        clusters.sort(key=lambda c: len(c), reverse=True)
        return clusters


# -------------------------------------------------------------------
# 실험용 dataclass / 설정
# -------------------------------------------------------------------

@dataclass
class GPSParams:
    eps_m: float
    min_samples: int


@dataclass
class EmbedParams:
    name: str
    extractor_factory: Callable[[], BaseDescriptorExtractor]
    similarity_threshold: float
    knn_k: int


@dataclass
class GeoParams:
    """
    max_features ↑ → 더 많은 키포인트, 더 정확하지만 느려짐.
    ratio_thresh ↓ (0.75 → 0.65) → 매칭 더 엄격, 매칭 수는 줄고 품질은 ↑.
    ransac_reproj_thresh ↓ (5.0 → 3.0) → inlier 판정 더 엄격, 구조가 조금 달라져도 실패하기 쉬움.
    min_good_matches ↑ → 텍스처 적은 장면엔 불리하지만, 안정된 매칭만 채택.
    geo_threshold ↑ → inlier 비율이 높은 쌍만 인정 → 고정밀, 저재현.
    """
    max_features: int
    ratio_thresh: float
    ransac_reproj_thresh: float
    min_good_matches: int
    geo_threshold: float


@dataclass
class ExperimentConfig:
    id: str
    gps: GPSParams
    embed: EmbedParams
    geo: GeoParams


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    n_photos: int
    n_gps_clusters: int
    n_scene_clusters: int
    metrics: Dict[str, float]
    stage_times: Dict[str, float]


# -------------------------------------------------------------------
# DeepCluster 생성 helper
# -------------------------------------------------------------------

def create_deep_cluster(embed: EmbedParams, geo: GeoParams, scorer: dict, enable_debug: bool) -> DeepCluster:
    descriptor = embed.extractor_factory()
    geo_matcher = LocalGeometryMatcher(
        max_features=geo.max_features,
        ratio_thresh=geo.ratio_thresh,
        ransac_reproj_thresh=geo.ransac_reproj_thresh,
        min_good_matches=geo.min_good_matches,
    )
    clusterer = DeepCluster(
        descriptor_extractor=descriptor,
        geo_matcher=geo_matcher,
        similarity_threshold=embed.similarity_threshold,
        geo_threshold=geo.geo_threshold,
        knn_k=embed.knn_k,
        min_cluster_size=2,
        structure_scores=scorer,
        enable_debug=enable_debug,
    )
    return clusterer


# -------------------------------------------------------------------
# Stage2~4: GPS 클러스터별 DeepCluster
# -------------------------------------------------------------------

def run_stage2_4_on_gps_clusters(
    photos: List[PhotoMeta],
    gps_labels: List[int],
    embed: EmbedParams,
    geo: GeoParams,
    scorer: dict,
    enable_debug: bool = True,
) -> Tuple[List[int], Dict[str, float]]:
    assert len(photos) == len(gps_labels)
    t0 = time.perf_counter()

    gps_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, lbl in enumerate(gps_labels):
        if lbl == -1:
            continue
        gps_to_indices[lbl].append(idx)

    final_labels = [-1] * len(photos)
    next_label = 0

    clusterer = create_deep_cluster(embed, geo, scorer, enable_debug)

    id_to_index: Dict[str, int] = {p.id: i for i, p in enumerate(photos)}
    
    pair_records: list[PairDebugRecord] = []
    for gps_lbl, idx_list in gps_to_indices.items():
        subset_photos = [photos[i] for i in idx_list]
        scene_clusters, pair_debug = clusterer.cluster(subset_photos)
        for scene in scene_clusters:
            if not scene:
                continue
            for p in scene:
                orig_idx = id_to_index[p.id]
                final_labels[orig_idx] = next_label
            next_label += 1
        
        for rec in pair_debug:
            pair_records.append(
                PairDebugRecord(
                    photo_path_i=rec["photo_path_i"],
                    photo_path_j=rec["photo_path_j"],
                    base_sim=rec["base_sim"],
                    eff_sim=rec["eff_sim"],
                    passed_sim=rec["passed_sim"],
                    geo_score=rec["geo_score"],
                    passed_geo=rec["passed_geo"],
                    gps_cluster_index=gps_lbl,
                )
            )

    t1 = time.perf_counter()
    stage_times = {"embedding+geo": t1 - t0}
    return final_labels, stage_times, pair_records


# -------------------------------------------------------------------
# 지표 계산
# -------------------------------------------------------------------

def compute_internal_metrics(
    features: np.ndarray,
    labels: List[int],
) -> Dict[str, float]:
    labels_arr = np.array(labels)
    mask = labels_arr >= 0
    if mask.sum() < 2 or features.shape[0] != labels_arr.shape[0]:
        return {}
    X = features[mask]
    y = labels_arr[mask]

    metrics: Dict[str, float] = {}
    try:
        metrics["silhouette"] = float(silhouette_score(X, y, metric="cosine"))
    except Exception:
        metrics["silhouette"] = float("nan")
    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(X, y))
    except Exception:
        metrics["davies_bouldin"] = float("nan")
    return metrics


def compute_external_metrics(
    true_labels: List[int],
    pred_labels: List[int],
) -> Dict[str, float]:
    t = np.array(true_labels)
    p = np.array(pred_labels)
    mask = t >= 0
    t = t[mask]
    p = p[mask]

    metrics: Dict[str, float] = {}
    metrics["ARI"] = float(adjusted_rand_score(t, p))
    metrics["NMI"] = float(normalized_mutual_info_score(t, p))
    return metrics


def _extract_features_for_metrics(
    photos: List[PhotoMeta],
    descriptor_extractor: BaseDescriptorExtractor,
) -> Tuple[np.ndarray, List[PhotoMeta]]:
    feats: List[np.ndarray] = []
    valid: List[PhotoMeta] = []
    for p in photos:
        v = descriptor_extractor.extract_one(p.path)
        if v is None:
            continue
        feats.append(v.astype(np.float32))
        valid.append(p)
    if not feats:
        return np.empty((0, 0), dtype=np.float32), []
    return np.stack(feats, axis=0), valid


# -------------------------------------------------------------------
# 시각화: 지도 / Mosaic
# -------------------------------------------------------------------

def plot_map_clusters(
    photos: List[PhotoMeta],
    labels: List[int],
    out_path: Path,
) -> None:
    labels_arr = np.array(labels)
    lat = np.array([p.lat for p in photos])
    lon = np.array([p.lon for p in photos])

    plt.figure(figsize=(8, 8))
    unique_labels = sorted(set(labels_arr))
    for lbl in unique_labels:
        mask = labels_arr == lbl
        if lbl == -1:
            plt.scatter(lon[mask], lat[mask], s=10, alpha=0.3, label="noise", marker="x")
        else:
            plt.scatter(lon[mask], lat[mask], s=20, alpha=0.7, label=f"cluster {lbl}")

    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.title("Cluster result on map")
    plt.legend(markerscale=1.5, fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def create_mosaic_for_experiment(
    exp_id: str,
    photos: List[PhotoMeta],
    labels: List[int],
    out_path: Path,
    thumb_size: Tuple[int, int] = (256, 256),
    max_photos_per_cluster: int = 5,
) -> None:
    labels_arr = np.array(labels)
    unique_labels = [lbl for lbl in sorted(set(labels_arr)) if lbl >= 0]
    if not unique_labels:
        logger.warning(f"[{exp_id}] no clusters to visualize.")
        return

    n_clusters = len(unique_labels)
    thumb_w, thumb_h = thumb_size
    cols = max_photos_per_cluster
    rows = n_clusters

    mosaic_w = cols * thumb_w
    mosaic_h = rows * thumb_h

    mosaic = Image.new("RGB", (mosaic_w, mosaic_h), color=(255, 255, 255))

    for row_idx, lbl in enumerate(unique_labels):
        indices = [i for i, l in enumerate(labels_arr) if l == lbl]
        cluster_photos = sorted(
            [photos[i] for i in indices],
            key=lambda p: (p.timestamp or datetime.min, str(p.id)),
        )
        cluster_photos = cluster_photos[:max_photos_per_cluster]

        for col_idx, p in enumerate(cluster_photos):
            try:
                img = Image.open(p.path).convert("RGB")
            except Exception:
                continue
            img = img.resize(thumb_size)
            x0 = col_idx * thumb_w
            y0 = row_idx * thumb_h
            mosaic.paste(img, (x0, y0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(out_path)


# -------------------------------------------------------------------
# summary.csv 기록
# -------------------------------------------------------------------

def write_summary_row(
    csv_path: Path,
    exp: ExperimentConfig,
    n_photos: int,
    n_gps_clusters: int,
    n_scene_clusters: int,
    metrics: Dict[str, float],
    stage_times: Dict[str, float],
) -> None:
    fieldnames = [
        "exp_id",
        "gps_eps_m",
        "gps_min_samples",
        "embed_name",
        "sim_th",
        "knn_k",
        "geo_max_features",
        "geo_ratio",
        "geo_ransac_thresh",
        "geo_min_matches",
        "geo_score_th",
        "n_photos",
        "n_gps_clusters",
        "n_scene_clusters",
        "time_gps_dbscan",
        "time_embedding_geo",
        "silhouette",
        "davies_bouldin",
        "ARI",
        "NMI",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {
            "exp_id": exp.id,
            "gps_eps_m": exp.gps.eps_m,
            "gps_min_samples": exp.gps.min_samples,
            "embed_name": exp.embed.name,
            "sim_th": exp.embed.similarity_threshold,
            "knn_k": exp.embed.knn_k,
            "geo_max_features": exp.geo.max_features,
            "geo_ratio": exp.geo.ratio_thresh,
            "geo_ransac_thresh": exp.geo.ransac_reproj_thresh,
            "geo_min_matches": exp.geo.min_good_matches,
            "geo_score_th": exp.geo.geo_threshold,
            "n_photos": n_photos,
            "n_gps_clusters": n_gps_clusters,
            "n_scene_clusters": n_scene_clusters,
            "time_gps_dbscan": stage_times.get("gps_dbscan", float("nan")),
            "time_embedding_geo": stage_times.get("embedding+geo", float("nan")),
            "silhouette": metrics.get("silhouette", float("nan")),
            "davies_bouldin": metrics.get("davies_bouldin", float("nan")),
            "ARI": metrics.get("ARI", float("nan")),
            "NMI": metrics.get("NMI", float("nan")),
        }
        writer.writerow(row)


def write_debug(pair_csv_path, pair_records):
    with pair_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "photo_path_i",
            "photo_path_j",
            "gps_cluster_index",
            "base_sim",
            "eff_sim",
            "passed_sim",
            "geo_score",
            "passed_geo",
        ])
        for r in pair_records:
            writer.writerow([
                r.photo_path_i,
                r.photo_path_j,
                r.gps_cluster_index,
                f"{r.base_sim:.6f}",
                f"{r.eff_sim:.6f}",
                int(r.passed_sim),
                "" if r.geo_score is None else f"{r.geo_score:.6f}",
                "" if r.passed_geo is None else int(r.passed_geo),
            ])

# -------------------------------------------------------------------
# 실험 1개 실행
# -------------------------------------------------------------------

def run_one_experiment(
    exp: ExperimentConfig,
    photos: List[PhotoMeta],
    true_labels: Optional[List[int]],
    base_out_dir: Path,
    scorer: dict,
) -> ExperimentResult:
    exp_dir = base_out_dir / exp.id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Stage1: GPS
    t0 = time.perf_counter()
    gps_labels = gps_dbscan_cluster(photos, eps_m=exp.gps.eps_m, min_samples=exp.gps.min_samples)
    t1 = time.perf_counter()
    stage_times: Dict[str, float] = {"gps_dbscan": t1 - t0}
    n_gps_clusters = len({l for l in gps_labels if l >= 0})

    # Stage2~4
    scene_labels, t_stage24, pair_records = run_stage2_4_on_gps_clusters(
        photos,
        gps_labels,
        exp.embed,
        exp.geo,
        scorer,
        enable_debug=True,  # 여기서 디버그 켬
    )

    stage_times.update(t_stage24)
    n_scene_clusters = len({l for l in scene_labels if l >= 0})
    n_photos = len(photos)

    # 내부 지표 위해 임베딩 재계산 (실제는 캐시 권장)
    descriptor_for_metrics = exp.embed.extractor_factory()
    feats, _ = _extract_features_for_metrics(photos, descriptor_for_metrics)
    internal_metrics = compute_internal_metrics(feats, scene_labels)
    metrics: Dict[str, float] = dict(internal_metrics)

    if true_labels is not None:
        external_metrics = compute_external_metrics(true_labels, scene_labels)
        metrics.update(external_metrics)

    # 시각화 저장
    plot_map_clusters(photos, scene_labels, out_path=exp_dir / f"{exp.id}_map.png")
    create_mosaic_for_experiment(exp.id, photos, scene_labels, out_path=exp_dir / f"{exp.id}_mosaic.png")

    # summary.csv
    summary_csv = base_out_dir / "summary.csv"
    write_summary_row(summary_csv, exp, n_photos, n_gps_clusters, n_scene_clusters, metrics, stage_times)

    pair_csv_path = base_out_dir / "pair_scores.csv"
    write_debug(pair_csv_path=pair_csv_path, pair_records=pair_records)

    return ExperimentResult(
        config=exp,
        n_photos=n_photos,
        n_gps_clusters=n_gps_clusters,
        n_scene_clusters=n_scene_clusters,
        metrics=metrics,
        stage_times=stage_times,
    )


# -------------------------------------------------------------------
# 전체 실험 grid 생성 + 실행 예시
# -------------------------------------------------------------------

def build_experiment_grid() -> List[ExperimentConfig]:
    gps_param_list = [
        GPSParams(eps_m=10.0, min_samples=3),
        # GPSParams(eps_m=8.0, min_samples=3),
        # GPSParams(eps_m=12.0, min_samples=3),
    ]

    def make_apgem() -> BaseDescriptorExtractor:
        return APGeMDescriptorExtractor(model_name="tf_efficientnet_b3_ns", image_size=320)

    def make_clip() -> BaseDescriptorExtractor:
        return CLIPDescriptorExtractor(model_name="ViT-B-32", pretrained="openai")

    def make_combined() -> BaseDescriptorExtractor:
        ap = APGeMDescriptorExtractor(model_name="tf_efficientnet_b3_ns", image_size=320)
        cl = CLIPDescriptorExtractor(model_name="ViT-B-32", pretrained="openai")
        return CombinedAPGeMCLIPExtractor(apgem=ap, clip=cl, w_apgem=0.8, w_clip=0.2)

    embed_param_list = [
        EmbedParams(name="APGeM", extractor_factory=make_apgem, similarity_threshold=0.7, knn_k=10),
        EmbedParams(name="CLIP", extractor_factory=make_clip, similarity_threshold=0.65, knn_k=10),
        EmbedParams(name="APGeM+CLIP", extractor_factory=make_combined, similarity_threshold=0.7, knn_k=10),
    ]

    geo_param_list = [
        GeoParams(
            max_features=1500,
            ratio_thresh=0.7,
            ransac_reproj_thresh=4.0,
            min_good_matches=10,
            geo_threshold=0.2,
        ),
        GeoParams(
            max_features=1000,
            ratio_thresh=0.72,
            ransac_reproj_thresh=4.0,
            min_good_matches=10,
            geo_threshold=0.25,
        ),
    ]

    configs: List[ExperimentConfig] = []
    idx = 0
    for gps in gps_param_list:
        for embed in embed_param_list:
            for geo in geo_param_list:
                exp_id = f"exp_{idx:03d}_{embed.name}_eps{gps.eps_m}_geo{geo.geo_threshold}"
                configs.append(ExperimentConfig(id=exp_id, gps=gps, embed=embed, geo=geo))
                idx += 1
    return configs


def run_all_experiments(
    photos: List[PhotoMeta],
    true_labels: Optional[List[int]],
    out_dir: Path,
) -> List[ExperimentResult]:
    configs = build_experiment_grid()
    scorer = compute_structure_scores_for_photos(photos=photos, scorer=StructureScorer())

    results: List[ExperimentResult] = []
    for exp in configs:
        logger.info(f"=== Running {exp.id} ===")
        res = run_one_experiment(exp, photos, true_labels, out_dir, scorer)
        results.append(res)
    return results


async def get_photos(image_dir):
    image_paths = [
        str(Path(image_dir) / fname)
        for fname in os.listdir(image_dir)
        if fname.lower().endswith(("png", "jpg", "jpeg"))
    ]
    ext = MetadataExtractor()
    tasks = [ext.extract(p) for p in image_paths]
    return await asyncio.gather(*tasks)


def compute_structure_scores_for_photos(
    photos: Sequence["PhotoMeta"],
    scorer: StructureScorer,
) -> Dict[str, float]:
    """
    PhotoMeta 리스트에 대해 구조물 점수를 0~1로 계산해서
    photo.id -> score 로 캐시.
    """
    scores: Dict[str, float] = {}
    for p in photos:
        try:
            s = scorer.score_from_path(p.path)
        except Exception as e:
            # 이미지 열기 실패 등 → 기본값 0.5 정도로 둬도 됨
            s = 0.5
        scores[p.original_name] = s
    return scores

# -------------------------------------------------------------------
# 사용 예시 (실제 프로젝트에서는 main에서 PhotoMeta 로딩 후 호출)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # 예시: photos 리스트와 true_labels 는 실제 코드에서 채워 넣어야 함
    photos: List[PhotoMeta] = []
    true_labels: Optional[List[int]] = None
    out_dir = Path("./.experiment/exp_results/6th")

    # 여기에 PhotoMeta 로딩 코드 추가 후 실행
    image_dir = "/Users/aohus/Workspaces/github/job-report-creator/backend/assets/어제/"
    photos = asyncio.run(get_photos(image_dir))
    true_labels = [int(photo.original_name.split("-")[0]) for photo in photos]
    logger.info(f"true labels: {true_labels}")
    results = run_all_experiments(photos, true_labels, out_dir)