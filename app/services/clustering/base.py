from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from common.models import PhotoMeta


class Clusterer(ABC):
    """Abstract base class for a clustering strategy."""

    @abstractmethod
    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        """
        Applies a clustering strategy to a list of photos.

        Args:
            photos: A list of PhotoMeta objects to cluster.

        Returns:
            A list of clusters, where each cluster is a list of PhotoMeta objects.
        """
        raise NotImplementedError()

    @staticmethod
    def condition(photos: Iterable[PhotoMeta]) -> bool:
        """Determines whether the strategy should be applied to the given photos."""
        return True

class BaseDescriptorExtractor(ABC):
    """Abstract base class for extracting image descriptors."""

    @abstractmethod
    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Extracts a descriptor (feature vector) for a single image.

        Args:
            image_path: The path to the image file.

        Returns:
            A numpy array representing the descriptor, or None if extraction fails.
        """
        raise NotImplementedError

class BaseGeometryMatcher(ABC):
    """Abstract base class for geometric matching between two images."""

    @abstractmethod
    def geo_score(self, path1: Path, path2: Path) -> float:
        """
        Computes a geometric similarity score between two images.

        Args:
            path1: The path to the first image.
            path2: The path to the second image.

        Returns:
            A float between 0.0 and 1.0, where 1.0 indicates high geometric similarity.
        """
        raise NotImplementedError
