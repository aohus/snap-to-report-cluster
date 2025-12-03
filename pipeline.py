import asyncio
import logging
import time
from typing import Dict, List

from domain.clusterers.base_clusterer import Clusterer
from domain.clusterers.gps_clusterer import GPSClusterer
from domain.clusterers.img_clusterer import DeepCluster
from domain.extractors.APGeM_extractor import APGeMDescriptorExtractor
from domain.extractors.metadata_extractor import MetadataExtractor
from domain.runner import ClusterRunner
from photometa import PhotoMeta

logger = logging.getLogger(__name__)


class PhotoClusteringPipeline:
    def __init__(self, CACHE_BASE):
        self.CACHE_BASE = CACHE_BASE
        self.metadata_extractor = MetadataExtractor()

        # Initialize clusterers
        clusterers = self._create_clusterers()
        self.clusterer = ClusterRunner(clusterers)
        logger.debug(f"Pipeline initialized with {len(clusterers)} clusterers.")

    def _create_clusterers(self) -> List[Clusterer]:
        """Factory method to create clustering clusterers based on config."""
        logger.debug("Creating clusterers...")
        # clusterer_map: Dict[str, Clusterer] = {"location": LocationClusterer()}
        apgem_extractor = APGeMDescriptorExtractor(
            model_name="tf_efficientnet_b3_ns",  # timm 모델 이름
            image_size=320,
            device=None,                         # None이면 mps→cuda→cpu 자동 선택
        )
        clusterer_map: Dict[str, Clusterer] = {
            "gps": GPSClusterer(),
            "deep_clusterer": DeepCluster(
                descriptor_extractor=apgem_extractor,
                similarity_threshold=0.8,
                geo_threshold=0.25,
                knn_k=5,
            )
        }

        active_clusterers = []
        for name in clusterer_map.keys():
            if name in clusterer_map:
                logger.debug(f"Activating clusterer: {name}")
                active_clusterers.append(clusterer_map[name])
            else:
                logger.warning(
                    f"Unknown clustering clusterer '{name}' in config. Ignoring."
                )
        return active_clusterers

    async def run(self, image_paths) -> list[list[PhotoMeta]]:
        logger.info("Pipeline run started")
        logger.info("Extracting metadata from images asynchronously...")
        tasks = [self.metadata_extractor.extract(p) for p in image_paths]
        photos = await asyncio.gather(*tasks)
        logger.info(f"Metadata extracted for {len(photos)} photos.")

        logger.info("Starting clustering pipeline...")
        final_scenes = await self.clusterer.process(photos)
        logger.info(
            f"Clustering pipeline finished. Final scene count: {len(final_scenes)}"
        )

        if not final_scenes:
            logger.warning("No scenes were generated. Skipping output generation.")
            return
        return final_scenes
