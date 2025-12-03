import logging
import math
from collections import Counter
from typing import List, Optional

from domain.clusterers.base_clusterer import Clusterer
from photometa import PhotoMeta
from pyproj import Geod

try:
    import hdbscan
except ImportError:
    hdbscan = None
    print("hdbscan not found, falling back to DBSCAN. For better results, please install it: pip install hdbscan")
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


class GPSClusterer(Clusterer):
    def __init__(self):
        self.executer = DbscanExecuter()
        self.max_dist_m = 15
        self.max_alt_diff_m = 15
        self.min_samples = 2

    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        return await self.cluster_loop(photos, is_main=True)
    
    def get_stats(self, clusters):
        if not clusters:
            return 0, 0
        counter = Counter([len(c) for c in clusters])
        return 3, 3
        
    async def cluster_loop(self, photos: List[PhotoMeta], is_main=False) -> List[List[PhotoMeta]]:
        while True:
            clusters = await self.executer.cluster(photos, 
                                                   self.max_dist_m, 
                                                   self.max_alt_diff_m, 
                                                   min_samples=self.min_samples)
            mode, mean = self.get_stats(clusters)
            logger.info(f"Iteration: photos: {len(photos)},mode={mode}, mean={mean}, max_dist_m={self.max_dist_m}, total clusters={len(clusters)}")
            
            # if is_main:
            #     self.executer.process_outlier(clusters, max_dist_m=self.max_dist_m, max_alt_diff_m=self.max_alt_diff_m, min_samples=self.min_samples)
            return clusters
            # if 3 <= mean <= 5 or self.max_dist_m <= 3 or self.max_dist_m >= self.config.MAX_LOCATION_DIST_M + 40.0:
            #     return clusters
            # if mean < 3:
            #     self.max_dist_m = self.max_dist_m + 2
            # if mean > 5:
            #     self.max_dist_m = self.max_dist_m - 2


class BaseExecuter:
    def __init__(self):
        self.geod = Geod(ellps="WGS84")

    def _haversine_distance_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        return self._distance_3d_m(lat1, lon1, lat2, lon2, None, None)
    
    def _geo_distanc_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        az12, az21, dist_geod = self.geod.inv(lon1, lat1, lon2, lat2)
        return dist_geod
    
    def _distance_3d_m(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        alt1: Optional[float] = None,
        alt2: Optional[float] = None,
    ) -> float:
        """
        위도/경도 + (옵션) 고도를 사용하는 3D 거리 (미터 단위).

        - alt1, alt2 둘 다 주어지면: 3D 거리 = sqrt(수평거리^2 + 고도차^2)
        - 하나라도 None이면: 수평 거리만 반환
        """
        R = 6371000.0  # Earth radius in meters

        # # 수평 거리 (haversine)
        # phi1 = math.radians(lat1)
        # phi2 = math.radians(lat2)
        # dphi = math.radians(lat2 - lat1)
        # dlambda = math.radians(lon2 - lon1)

        # a = (
        #     math.sin(dphi / 2.0) ** 2
        #     + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        # )
        # c = 2 * math.asin(math.sqrt(a))
        # d2d = R * c  # 수평 거리

        d2d = self._geo_distanc_m(lat1, lon1, lat2, lon2)
        
        # 고도 포함 3D 거리
        w_alt = 0.3
        if alt1 is not None and alt2 is not None:
            dz = alt2 - alt1
            return math.sqrt(d2d**2 + (w_alt * dz)**2)
        else:
            return d2d
    
    async def cluster(self, photos: List[PhotoMeta], max_dist_m, max_alt_diff_m=None, *, args, kwargs) -> List[List[PhotoMeta]]:
        raise NotImplementedError
    
    async def process_outlier(self, clusters: List[List[PhotoMeta]], max_dist_me, max_alt_diff_m=None, *, args, kwargs) -> List[List[PhotoMeta]]:
        raise NotImplementedError


class GreedyExecuter(BaseExecuter):
    async def cluster(self, photos: List[PhotoMeta], max_dist_m, max_alt_diff_m=None, min_samples=None) -> List[List[PhotoMeta]]:
        """Clusters photos based on their GPS location."""
        clusters: List[List[PhotoMeta]] = []

        # Filter out photos without GPS data first
        photos_with_gps = [p for p in photos if p.lat is not None and p.lon is not None]

        for photo in photos_with_gps:
            assigned = False
            for cluster in clusters:
                center = cluster[0]
                
                dist = self._haversine_distance_m(photo.lat, photo.lon, center.lat, center.lon)
                
                alt_diff = 0.0
                if photo.alt is not None and center.alt is not None:
                    alt_diff = abs(photo.alt - center.alt)

                if dist <= self.max_dist_m and alt_diff <= self.max_alt_diff_m:
                    cluster.append(photo)
                    assigned = True
                    break
            
            if not assigned:
                clusters.append([photo])
        
        # Include photos without GPS data as their own individual clusters
        photos_without_gps = [p for p in photos if p.lat is None or p.lon is None]
        for photo in photos_without_gps:
            clusters.append([photo])
        return clusters
    
    async def process_outlier(self, clusters: List[List[PhotoMeta]], max_dist_m, max_alt_diff_m=None, min_samples=None) -> List[List[PhotoMeta]]:
        final_clusters = []
        single_photo_cluster = []

        for cluster in clusters:
            if len(cluster) < 2:
                single_photo_cluster.append(cluster[0])
            if len(cluster) > 5:
                sub_clusters = await self.split_large_clusters(cluster, max_dist_m=max_dist_m)
                for sub_cluster in sub_clusters:
                    if len(sub_cluster) == 1:
                        single_photo_cluster.append(sub_cluster[0])
                    else:
                        final_clusters.append(sub_cluster)
            else:
                final_clusters.append(cluster)
        
        if single_photo_cluster:
            sub_clusters = await self.gather_single_clusters(single_photo_cluster, max_dist_m=max_dist_m)
            single_photo_cluster = []
            for sub_cluster in sub_clusters:
                if len(sub_cluster) == 1:
                    single_photo_cluster.append(sub_cluster[0])
            final_clusters.append(single_photo_cluster)
        return final_clusters
    
    async def split_large_clusters(self, cluster: List[PhotoMeta], max_dist_m: float = None) -> List[List[PhotoMeta]]:
        if len(cluster) > 10:
            self.max_dist_m = max_dist_m - max_dist_m / 4.0
        return await self.cluster(cluster)

    async def gather_single_clusters(self, cluster: List[PhotoMeta], max_dist_m: float = None) -> List[PhotoMeta]:
        self.max_dist_m = max_dist_m + max_dist_m / 5.0
        return await self.cluster(cluster)


class DbscanExecuter(BaseExecuter):
    def build_distance_matrix(self, photos):
        gps_photos = [p for p in photos if p.lat is not None and p.lon is not None]
        n = len(gps_photos)
        if n == 0:
            return gps_photos, None

        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = self._haversine_distance_m(
                    gps_photos[i].lat, gps_photos[i].lon,
                    gps_photos[j].lat, gps_photos[j].lon,
                )
                D[i, j] = D[j, i] = d
        return gps_photos, D

    async def cluster(self, photos, max_dist_m: float, max_alt_diff_m: float, min_samples: int = 3):
        gps_photos, D = self.build_distance_matrix(photos)
        if D is None:
            return []

        db = DBSCAN(
            eps=max_dist_m,          # 여기서는 그대로 "미터" 단위
            min_samples=min_samples,
            metric="precomputed",
        ).fit(D)

        labels = db.labels_
        clusters_dict = {}
        for label, p in zip(labels, gps_photos):
            # if label == -1:
            #     continue
            clusters_dict.setdefault(label, []).append(p)
        clusters = [clusters_dict[k] for k in sorted(clusters_dict.keys())]
        return clusters

    # async def process_outlier(self, clusters: List[List[PhotoMeta]], max_dist_m, max_alt_diff_m, min_samples) -> List[List[PhotoMeta]]:
    #     final_clusters = []
    #     for cluster in clusters:
    #         if len(cluster) > 5:
    #             sub_clusters = await self.cluster(cluster, max_dist_m, max_alt_diff_m, min_samples)
    #             for sub_cluster in sub_clusters:
    #                 final_clusters.append(sub_cluster)
    #         else:
    #             final_clusters.append(cluster)
    #     return final_clusters