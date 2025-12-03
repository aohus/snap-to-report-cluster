#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
deep_cluster.py

Stage 2~5 구현:

- Stage 2: 전역 임베딩 추출 (place-recognition 용도)
- Stage 3: SIFT + RANSAC 로 기하 검증
- Stage 4: k-NN + 연결요소 기반 클러스터링
- Stage 5: 각 클러스터 내부 시간 순 정렬 (전/중/후 등 후처리용)

사용 예:
    from deep_cluster import DeepCluster, PhotoMeta

    photos = [
        PhotoMeta(id=1, path="/path/a.jpg", timestamp=...),
        PhotoMeta(id=2, path="/path/b.jpg", timestamp=...),
        ...
    ]
    clusterer = DeepCluster()
    clusters = clusterer.cluster(photos)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from domain.extractors.APGeM_extractor import APGeMDescriptorExtractor
from photometa import PhotoMeta
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# -------------------------------------------------------------------
# 로컬 특징 + RANSAC 기반 기하 검증
# -------------------------------------------------------------------

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


class LocalGeometryMatcher:
    """
    두 이미지가 '정말 같은 장면'인지 판단하기 위한 기하 검증기.
    SIFT + Lowe ratio test + RANSAC (homography 또는 fundamental matrix) 사용.
    """

    def __init__(
        self,
        max_features: int = 1500,
        ratio_thresh: float = 0.75,
        ransac_reproj_thresh: float = 5.0,
        min_good_matches: int = 15,
    ) -> None:
        self.max_features = max_features
        self.ratio_thresh = ratio_thresh
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.min_good_matches = min_good_matches

        if cv2 is None:
            logger.warning(
                "⚠️ OpenCV(cv2)가 설치되어 있지 않습니다. "
                "기하 검증 단계는 항상 score=1.0 을 반환합니다."
            )
            self.enabled = False
            self.detector = None
        else:
            self.enabled = True
            # SIFT 또는 AKAZE 등으로 교체 가능
            try:
                self.detector = cv2.SIFT_create(nfeatures=self.max_features)  # type: ignore[attr-defined]
            except Exception:
                logger.warning(
                    "⚠️ SIFT 생성 실패. opencv-contrib-python 이 필요합니다. "
                    "기하 검증 단계는 비활성화됩니다."
                )
                self.detector = None
                self.enabled = False

    def _load_gray(self, path: Path) -> Optional[np.ndarray]:
        if not self.enabled or self.detector is None:
            return None
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # type: ignore[attr-defined]
        if img is None:
            logger.warning(f"⚠️ Failed to read image for geometry: {path}")
            return None
        return img

    def geo_score(self, path1: Path, path2: Path) -> float:
        """
        0.0 ~ 1.0 사이의 기하학적 일관성 점수.
        - 0.0 에 가까울수록 구조가 다르거나 매칭 실패
        - 1.0 에 가까울수록 구조가 상당히 일치

        기하 검증을 사용하지 못하는 환경(cv2 없음 등)에서는 항상 1.0 반환.
        """
        if not self.enabled or self.detector is None:
            return 1.0  # fallback: 기하 검증을 생략하고 항상 통과

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

        # Homography + RANSAC
        H, mask = cv2.findHomography(  # type: ignore[attr-defined]
            pts1,
            pts2,
            cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_thresh,  # type: ignore[attr-defined]
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
# 연결요소 (Union-Find)
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
# DeepCluster
# -------------------------------------------------------------------

class DeepCluster:
    """
    Stage 2~5 클러스터링 구현.

    - 입력: 동일 GPS 클러스터(또는 job 단위)로 묶인 PhotoMeta 리스트
    - 출력: PhotoMeta 의 리스트 리스트 (각 리스트 = 같은 장면 클러스터)

    interface requirement:
        def cluster(self, photos: Sequence[PhotoMeta]) -> List[List[PhotoMeta]]
    """

    def __init__(
        self,
        descriptor_extractor: Optional[APGeMDescriptorExtractor] = None,
        geo_matcher: Optional[LocalGeometryMatcher] = None,
        similarity_threshold: float = 0.7,
        geo_threshold: float = 0.25,
        min_cluster_size: int = 2,
        knn_k: int = 10,
    ) -> None:
        """
        Args:
            descriptor_extractor: 전역 임베딩 추출기. None이면 기본 CLIP extractor 사용.
            geo_matcher: 기하 검증기. None이면 기본 LocalGeometryMatcher 사용.
            similarity_threshold: 전역 임베딩 코사인 유사도 임계값.
            geo_threshold: 기하 score 임계값 (0~1).
            min_cluster_size: 이보다 작은 클러스터는 버리거나 단독 처리.
            knn_k: k-NN 그래프에서 이웃 개수.
        """
        self.descriptor_extractor = (
            descriptor_extractor if descriptor_extractor is not None else APGeMDescriptorExtractor()
        )
        self.geo_matcher = geo_matcher if geo_matcher is not None else LocalGeometryMatcher()
        self.similarity_threshold = float(similarity_threshold)
        self.geo_threshold = float(geo_threshold)
        self.min_cluster_size = int(min_cluster_size)
        self.knn_k = int(knn_k)

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------
    async def cluster(self, photos: Sequence[PhotoMeta]) -> List[List[PhotoMeta]]:
        """
        Stage 2~5 전체 파이프라인.

        1) 전역 임베딩 추출 (Stage 2)
        2) k-NN 그래프 생성 (Stage 2)
        3) SIFT+RANSAC 기하 검증으로 간선 필터링 (Stage 3)
        4) Union-Find로 연결요소 → 클러스터 (Stage 4)
        5) 각 클러스터 내부를 촬영 시각 순으로 정렬 (Stage 5)

        Returns:
            List[List[PhotoMeta]]: 각 내부 리스트가 "같은 장면" 클러스터.
        """
        if not photos:
            return []

        logger.info(f"📷 DeepCluster: {len(photos)} photos 입력")

        # Stage 2: 전역 임베딩
        features, valid_photos = self._extract_features(photos)
        if len(valid_photos) < 2:
            logger.warning("⚠️ usable photo 가 2장 미만입니다. 클러스터링 불가.")
            return [[p] for p in valid_photos]

        # Stage 2: k-NN 그래프 (전역 유사도 기준)
        edges = self._build_candidate_edges(features, valid_photos)

        # Stage 3: 기하 검증 + edge 필터링
        edges = self._filter_edges_by_geometry(edges, valid_photos)

        # Stage 4: Union-Find 로 연결요소
        labels = self._connected_components(len(valid_photos), edges)

        # Stage 5: 라벨별로 묶고, 시간 순 정렬
        clusters = self._build_clusters_from_labels(labels, valid_photos)

        logger.info(
            f"✅ DeepCluster 완료: {len(clusters)} clusters, "
            f"{sum(len(c) for c in clusters)} photos."
        )
        return clusters

    # ------------------------------------------------------------------
    # Stage 2: feature extraction
    # ------------------------------------------------------------------
    def _extract_features(
        self, photos: Sequence[PhotoMeta]
    ) -> Tuple[np.ndarray, List[PhotoMeta]]:
        feats: List[np.ndarray] = []
        valid_photos: List[PhotoMeta] = []

        for p in photos:
            path = Path(p.path)
            if not path.is_file():
                logger.warning(f"⚠️ 이미지 파일이 존재하지 않습니다: {path}")
                continue
            vec = self.descriptor_extractor.extract_one(path)
            if vec is None:
                continue
            feats.append(vec)
            valid_photos.append(p)

        if not feats:
            return np.empty((0, 0), dtype=np.float32), []

        features_array = np.stack(feats, axis=0)
        logger.info(f"📊 feature shape: {features_array.shape}")
        return features_array, valid_photos

    # ------------------------------------------------------------------
    # Stage 2: k-NN graph construction
    # ------------------------------------------------------------------
    def _build_candidate_edges(
        self,
        features: np.ndarray,
        photos: List[PhotoMeta],
    ) -> List[Tuple[int, int]]:
        n, d = features.shape
        if n == 0:
            return []

        k = min(max(2, self.knn_k), n)
        logger.info(f"🔗 k-NN graph 구성 (n={n}, d={d}, k={k})")

        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(features)
        distances, indices = nn.kneighbors(features)

        edges: List[Tuple[int, int]] = []

        for i in range(n):
            # indices[i][0] == 자기 자신
            for dist, j in zip(distances[i][1:], indices[i][1:]):
                sim = 1.0 - float(dist)
                if sim < self.similarity_threshold:
                    continue
                # 아직 기하 검증은 하지 않고, 후보로만 저장 (Stage 3에서 필터링)
                if i < j:
                    edges.append((i, j))

        logger.info(f"🔍 전역 임베딩 기준 후보 edge 수: {len(edges)}")
        return edges

    # ------------------------------------------------------------------
    # Stage 3: geometric verification
    # ------------------------------------------------------------------
    def _filter_edges_by_geometry(
        self,
        edges: List[Tuple[int, int]],
        photos: List[PhotoMeta],
    ) -> List[Tuple[int, int]]:
        if not edges:
            return []

        logger.info("🧮 SIFT + RANSAC 기하 검증 시작")
        kept: List[Tuple[int, int]] = []

        for (i, j) in edges:
            path_i = Path(photos[i].path)
            path_j = Path(photos[j].path)
            score_geo = self.geo_matcher.geo_score(path_i, path_j)
            if score_geo >= self.geo_threshold:
                kept.append((i, j))

        logger.info(
            f"📌 기하 검증 통과 edge 수: {len(kept)} "
            f"(원래 {len(edges)} 개 중)"
        )
        return kept

    # ------------------------------------------------------------------
    # Stage 4: connected components
    # ------------------------------------------------------------------
    def _connected_components(
        self,
        n: int,
        edges: List[Tuple[int, int]],
    ) -> List[int]:
        if n == 0:
            return []

        if not edges:
            # edge가 없으면 각자 singleton cluster
            return list(range(n))

        uf = UnionFind(n)
        for i, j in edges:
            uf.union(i, j)
        labels = uf.to_labels()
        return labels

    # ------------------------------------------------------------------
    # Stage 5: build clusters + sort by time
    # ------------------------------------------------------------------
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
            # min_cluster_size 보다 작은 cluster 는 일단 포함은 하되,
            # 필요하면 여기서 필터링할 수 있음.
            cluster = sorted(
                items,
                key=lambda p: (p.timestamp or datetime.min, str(p.id)),
            )
            clusters.append(cluster)

        # 클러스터 크기 큰 순으로 정렬 (optional)
        clusters.sort(key=lambda c: len(c), reverse=True)
        return clusters
    
    def condition(self, c):
        return True