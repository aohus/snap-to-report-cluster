from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from common.models import PhotoMeta
from config import ClusteringConfig
from services.clustering.base import (
    BaseDescriptorExtractor,
    BaseGeometryMatcher,
    Clusterer,
)
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


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


class ImageClusterer(Clusterer):
    """
    Stage 2~5 클러스터링 구현.

    - 입력: 동일 GPS 클러스터(또는 job 단위)로 묶인 PhotoMeta 리스트
    - 출력: PhotoMeta 의 리스트 리스트 (각 리스트 = 같은 장면 클러스터)
    """

    def __init__(
        self,
        config: ClusteringConfig,
        descriptor_extractor: BaseDescriptorExtractor,
        geo_matcher: BaseGeometryMatcher,
    ) -> None:
        self.config = config.deep_cluster
        self.descriptor_extractor = descriptor_extractor
        self.geo_matcher = geo_matcher
        self.similarity_threshold = config.descriptor.similarity_threshold
        self.geo_threshold = self.config.geo_threshold
        self.min_cluster_size = self.config.min_cluster_size
        self.knn_k = config.descriptor.knn_k

    async def cluster(self, photos: Sequence[PhotoMeta]) -> List[List[PhotoMeta]]:
        """
        Stage 2~5 전체 파이프라인.

        1) 전역 임베딩 추출 (Stage 2)
        2) k-NN 그래프 생성 (Stage 2)
        3) 기하 검증으로 간선 필터링 (Stage 3)
        4) Union-Find로 연결요소 → 클러스터 (Stage 4)
        5) 각 클러스터 내부를 촬영 시각 순으로 정렬 (Stage 5)

        Returns:
            List[List[PhotoMeta]]: 각 내부 리스트가 "같은 장면" 클러스터.
        """
        if not photos:
            return []

        logger.info(f"📷 ImageClusterer: {len(photos)} photos 입력")

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
            f"✅ ImageClusterer 완료: {len(clusters)} clusters, "
            f"{sum(len(c) for c in clusters)} photos."
        )
        return clusters

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
            
            # Extract features with metadata
            metadata = {'lat': p.lat, 'lon': p.lon, 'timestamp': p.timestamp}
            vec = self.descriptor_extractor.extract_one(path, metadata=metadata)
            
            if vec is None:
                continue
            feats.append(vec)
            valid_photos.append(p)

        if not feats:
            return np.empty((0, 0), dtype=np.float32), []

        features_array = np.stack(feats, axis=0)
        logger.info(f"📊 feature shape: {features_array.shape}")
        return features_array, valid_photos

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
            for dist, j in zip(distances[i][1:], indices[i][1:]):
                sim = 1.0 - float(dist)
                if sim < self.similarity_threshold:
                    continue
                if i < j:
                    edges.append((i, j))

        logger.info(f"🔍 전역 임베딩 기준 후보 edge 수: {len(edges)}")
        return edges

    def _filter_edges_by_geometry(
        self,
        edges: List[Tuple[int, int]],
        photos: List[PhotoMeta],
    ) -> List[Tuple[int, int]]:
        if not edges:
            return []

        logger.info("🧮 기하 검증 시작")
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

    def _connected_components(
        self,
        n: int,
        edges: List[Tuple[int, int]],
    ) -> List[int]:
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
            # min_cluster_size 보다 작은 cluster 는 일단 포함은 하되,
            # 필요하면 여기서 필터링할 수 있음.
            cluster = sorted(
                items,
                key=lambda p: (p.timestamp or datetime.min, str(p.id)),
            )
            clusters.append(cluster)

        clusters.sort(key=lambda c: len(c), reverse=True)
        return clusters

    def condition(self, cluster):
        return len(cluster) > 4