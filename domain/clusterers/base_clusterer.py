from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional

from photometa import PhotoMeta


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

    def _validate_inputs(
        photos: Optional[Iterable[PhotoMeta]],
        input_dir: Optional[str]
    ) -> None:
        if not input_dir and not photos:
            raise ValueError("input_dir 또는 path_list 중 하나는 반드시 지정해야 합니다.")

    @staticmethod
    def condition(photos: Iterable[PhotoMeta]) -> bool:
        """ Determines whether the strategy should be applied to the given photos. """
        return True
