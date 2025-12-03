import logging
import math
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from pyproj import Geod

from app_v2.common.models import PhotoMeta
from app_v2.config import ClusteringConfig
from app_v2.services.clustering.base import Clusterer

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
            # If no photos with GPS, each photo without GPS forms its own cluster
            return [[p] for p in photos_without_gps]

        # Calculate a reference point (mean of all GPS photos)
        lat0 = sum(p.lat for p in photos_with_gps) / len(photos_with_gps)
        lon0 = sum(p.lon for p in photos_with_gps) / len(photos_with_gps)

        # Convert lat/lon to local x,y coordinates in meters
        coords = np.array(
            [latlon_to_xy_m(p.lat, p.lon, lat0, lon0) for p in photos_with_gps],
            dtype=np.float32,
        )

        # Apply DBSCAN
        db = DBSCAN(eps=self.config.eps_m, min_samples=self.config.min_samples, metric="euclidean")
        labels = db.fit_predict(coords)

        # Group photos by cluster label
        clusters_dict = defaultdict(list)
        for label, photo in zip(labels, photos_with_gps):
            clusters_dict[label].append(photo)
        
        # Format clusters, excluding noise (-1 label)
        gps_clusters = [sorted(clusters_dict[lbl], key=lambda p: p.timestamp or 0) for lbl in sorted(clusters_dict.keys()) if lbl != -1]
        
        # Add noise points as individual clusters
        if -1 in clusters_dict:
            gps_clusters.extend([[p] for p in clusters_dict[-1]])
        
        # Add photos without GPS as individual clusters
        gps_clusters.extend([[p] for p in photos_without_gps])

        logger.info(f"GPS clustering resulted in {len(gps_clusters)} clusters from {len(photos)} photos.")
        return gps_clusters
