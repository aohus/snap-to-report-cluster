import logging
from typing import List

from domain.clusterers.base_clusterer import Clusterer
from photometa import PhotoMeta

logger = logging.getLogger(__name__)


class ClusterRunner:
    def __init__(self, clusterers: List[Clusterer]):
        self.clusterers = clusterers

    async def process(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        """
        Processes photos by applying a series of clustering clusterers in sequence.
        """
        # Start with a single cluster containing all photos
        clusters = [photos]
        
        for clusterer in self.clusterers:
            logger.info(f"Applying clusterer: {clusterer.__class__.__name__}")
            
            new_clusters = []
            # Apply the clusterer to each existing cluster
            for cluster in clusters:
                if not cluster:
                    continue
                if clusterer.condition(cluster):
                    sub_clusters = await clusterer.cluster(cluster)
                    new_clusters.extend(sub_clusters)
                else:
                    new_clusters.append(cluster)
            
            clusters = new_clusters
            logger.info(f"Resulted in {len(clusters)} clusters.")
        return clusters
