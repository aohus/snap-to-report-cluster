import logging
import math
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
from common.models import PhotoMeta
from config import ClusteringConfig
from pyproj import Geod
from services.clustering.base import Clusterer
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6371000.0  # meters

def latlon_to_xy_m(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)
    x = (lon_rad - lon0_rad) * math.cos((lat_rad + lat0_rad) / 2.0) * EARTH_RADIUS_M
    y = (lat_rad - lat0_rad) * EARTH_RADIUS_M
    return x, y


class GPSClusterer(Clusterer):
    def __init__(self, config: ClusteringConfig):
        self.config = config.gps
        self.geod = Geod(ellps="WGS84") # Initialize Geod for _geo_distance_m if needed

    def _geo_distance_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates geodesic distance between two points."""
        az12, az21, dist_geod = self.geod.inv(lon1, lat1, lon2, lat2)
        return dist_geod

    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        if not photos:
            return []

        # Filter out photos without GPS data
        photos_with_gps = [p for p in photos if p.lat is not None and p.lon is not None]
        photos_without_gps = [p for p in photos if p.lat is None or p.lon is None]

        if not photos_with_gps:
            return [[p] for p in photos_without_gps]

        lat0 = sum(p.lat for p in photos_with_gps) / len(photos_with_gps)
        lon0 = sum(p.lon for p in photos_with_gps) / len(photos_with_gps)

        coords = np.array(
            [latlon_to_xy_m(p.lat, p.lon, lat0, lon0) for p in photos_with_gps],
            dtype=np.float32,
        )

        # Optimize eps to get cluster sizes mostly around 3
        best_labels = self._optimize_clustering(coords)
        
        # Group photos by cluster label
        clusters_dict = defaultdict(list)
        for label, photo in zip(best_labels, photos_with_gps):
            clusters_dict[label].append(photo)

        # Assign noise (-1) to nearest clusters
        if -1 in clusters_dict:
            noise_photos = clusters_dict.pop(-1)
            
            # If no valid clusters exist, group all noise into one cluster (or handle as needed)
            if not clusters_dict:
                clusters_dict[0] = noise_photos
            else:
                # Calculate centroids of existing clusters
                centroids = {}
                for label, photos in clusters_dict.items():
                    c_lat = sum(p.lat for p in photos) / len(photos)
                    c_lon = sum(p.lon for p in photos) / len(photos)
                    centroids[label] = (c_lat, c_lon)
                
                # Assign each noise photo to the nearest centroid
                for p in noise_photos:
                    best_label = None
                    min_dist = float('inf')
                    for label, (c_lat, c_lon) in centroids.items():
                        dist = self._geo_distance_m(p.lat, p.lon, c_lat, c_lon)
                        if dist < min_dist:
                            min_dist = dist
                            best_label = label
                    if best_label is not None:
                        clusters_dict[best_label].append(p)
        
        gps_clusters = [sorted(photos, key=lambda p: p.timestamp or 0) for photos in clusters_dict.values()]
        if photos_without_gps:
            gps_clusters.extend([[p] for p in photos_without_gps])

        logger.info(f"GPS clustering resulted in {len(gps_clusters)} clusters from {len(photos)} photos.")
        return gps_clusters

    def _optimize_clustering(self, coords: np.ndarray) -> np.ndarray:
        """
        Try multiple eps values to find the one that produces cluster sizes closest to 3.
        min_samples is fixed to 3 to avoid small clusters (1, 2).
        """
        eps_candidates = [4, 8, 10, 20, 30, 50]
        best_score = -float('inf')
        best_labels = None
        
        # If N < 3, we can't make clusters of 3. DBSCAN with min_samples=3 returns all noise.
        # We'll handle the "all noise" case in the caller by merging.
        if len(coords) < 3:
            return np.zeros(len(coords), dtype=int) # All in cluster 0

        for eps in eps_candidates:
            db = DBSCAN(eps=eps, min_samples=3, metric="euclidean")
            labels = db.fit_predict(coords)
            
            # Calculate score
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            
            noise_count = np.sum(labels == -1)
            total_count = len(labels)
            
            if not unique_labels:
                # All noise. Bad score.
                score = -1000 
            else:
                cluster_sizes = [np.sum(labels == l) for l in unique_labels]
                # Use median to be robust against outliers (one huge cluster)
                median_size = np.median(cluster_sizes)
                
                # Deviation from 3
                size_score = -abs(median_size - 3)
                
                # Noise penalty.
                noise_score = -(noise_count / total_count) * 5
                
                score = size_score + noise_score
            
            if best_labels is None or score > best_score:
                best_score = score
                best_labels = labels
        
        return best_labels
