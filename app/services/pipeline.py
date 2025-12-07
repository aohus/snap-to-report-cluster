import asyncio
import logging
import time
from typing import Dict, List, Type

from common.models import PhotoMeta
from config import ClusteringConfig
from services.clustering.base import (
    BaseDescriptorExtractor,
    BaseGeometryMatcher,
    Clusterer,
)
from services.clustering.descriptors import (
    APGeMDescriptorExtractor,
    CLIPDescriptorExtractor,
    CombinedAPGeMCLIPExtractor,
)
from services.clustering.finetuned_descriptors import FinetunedDescriptorExtractor
from services.clustering.geometry import LoFTRMatcher, SIFTMatcher
from services.clustering.gps import GPSClusterer
from services.clustering.image import ImageClusterer
from services.clustering.people_detector import create_people_detector
from services.clustering.segmenter import SemanticSegmenter
from services.metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)


class ClusterRunner:
    """
    Applies a series of clustering strategies in sequence.
    """
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
                # The condition method determines if a clusterer should be applied to a specific sub-cluster.
                # In this refactor, for simplicity, we assume all clusterers apply to all sub-clusters,
                # but this can be extended.
                if clusterer.condition(cluster):
                    sub_clusters = await clusterer.cluster(cluster)
                    new_clusters.extend(sub_clusters)
                else:
                    new_clusters.append(cluster)
            
            clusters = new_clusters
            logger.info(f"Resulted in {len(clusters)} clusters after {clusterer.__class__.__name__}.")
        return clusters


class PhotoClusteringPipeline:
    def __init__(self, config: ClusteringConfig, cache_base_dir: str = ".cache"):
        self.config = config
        self.cache_base_dir = cache_base_dir # Not directly used in this version but kept for potential future use
        
        self.metadata_extractor = MetadataExtractor()

        # Initialize common components
        self.people_detector = None
        if self.config.remove_people:
            self.people_detector = create_people_detector()

        self.semantic_segmenter = None
        if self.config.use_semantic_mask_for_loftr:
            self.semantic_segmenter = SemanticSegmenter(config)

        # Initialize descriptor extractor
        self.descriptor_extractor = self._create_descriptor_extractor()

        # Initialize geometry matcher
        self.geo_matcher = self._create_geometry_matcher()

        # Initialize clusterers
        clusterers = self._create_clusterers()
        self.cluster_runner = ClusterRunner(clusterers)
        logger.debug(f"Pipeline initialized with {len(clusterers)} clusterers.")

    def _create_descriptor_extractor(self) -> BaseDescriptorExtractor:
        # Use finetuned multi-modal extractor
        return FinetunedDescriptorExtractor(
            self.config,
            people_detector=self.people_detector
        )

    def _create_geometry_matcher(self) -> BaseGeometryMatcher:
        # Default to LoFTR as it was the focus in the semantic experiment
        # Can be made configurable
        return LoFTRMatcher(self.config, segmenter=self.semantic_segmenter)
    
    def _create_clusterers(self) -> List[Clusterer]:
        """Factory method to create clustering clusterers based on config."""
        logger.debug("Creating clusterers...")
        
        # GPS Clusterer (Stage 1)
        gps_clusterer = GPSClusterer(self.config)

        # Image Clusterer (Stages 2-5)
        image_clusterer = ImageClusterer(
            self.config,
            descriptor_extractor=self.descriptor_extractor,
            geo_matcher=self.geo_matcher
        )

        return [gps_clusterer, image_clusterer]

    async def run(self, image_paths: List[str]) -> List[List[PhotoMeta]]:
        logger.info("Pipeline run started")
        logger.info("Extracting metadata from images asynchronously...")
        tasks = [self.metadata_extractor.extract(p) for p in image_paths]
        photos = await asyncio.gather(*tasks)
        logger.info(f"Metadata extracted for {len(photos)} photos.")

        logger.info("Starting clustering pipeline...")
        final_clusters = await self.cluster_runner.process(photos)
        logger.info(
            f"Clustering pipeline finished. Final cluster count: {len(final_clusters)}"
        )

        if not final_clusters:
            logger.warning("No clusters were generated. Skipping output generation.")
            return []
        return final_clusters
