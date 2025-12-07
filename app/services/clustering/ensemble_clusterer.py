import logging
from typing import Dict, List

import numpy as np

try:
    import hdbscan
except ImportError:
    hdbscan = None
    print("hdbscan not found, falling back to DBSCAN. For better results, please install it: pip install hdbscan")
from app.domain.clusterers.base_clusterer import Clusterer
from app.domain.clusterers.deep_clusterer import DeepClusterer
from app.models.photometa import PhotoMeta
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

class EnsembleClusterer(Clusterer):
    """
    An advanced clusterer that uses an ensemble of deep learning features
    and the HDBSCAN algorithm for more robust clustering.
    """
    def __init__(self, deep_clusterer: DeepClusterer, min_cluster_size: int = 2, min_samples: int = 1):
        """
        Initializes the EnsembleClusterer.

        Args:
            deep_clusterer: An instance of DeepClusterer to extract features.
            min_cluster_size: The minimum size of a cluster for HDBSCAN.
            min_samples: The number of samples in a neighborhood for a point to be considered as a core point for HDBSCAN.
        """
        if hdbscan is None:
            logger.warning("HDBSCAN is not installed. Falling back to DBSCAN which might produce lower quality clusters.")
        
        self.deep_clusterer = deep_clusterer
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        logger.debug("EnsembleClusterer initialized.")


    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        """
        Clusters photos using an ensemble of features and HDBSCAN.
        """
        if not photos:
            return []
            
        if len(photos) < self.min_cluster_size:
            logger.info(f"Skipping clustering for {len(photos)} photos (less than min_cluster_size).")
            return [photos]

        logger.info(f"Starting ensemble clustering for {len(photos)} photos.")

        # 1. Extract features for all photos using the deep_clusterer's cache mechanism
        features_dict_list = await self._extract_all_features(photos)

        if not features_dict_list:
            logger.warning("Could not extract features for any photos. Aborting clustering.")
            return [photos] # Return as a single group if no features found

        # 2. Create ensemble feature vectors
        ensemble_features = self._create_ensemble_vectors(features_dict_list)
        
        logger.info(f"Ensemble feature vectors created with shape: {ensemble_features.shape}")

        # 3. Perform clustering using HDBSCAN
        if hdbscan:
            logger.debug(f"Using HDBSCAN with min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples}")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, metric='euclidean')
            labels = clusterer.fit_predict(ensemble_features)
            logger.info(f"Clustering complete. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters with HDBSCAN.")
        else:
            # Fallback to DBSCAN if hdbscan is not available
            # Note: DBSCAN is less effective here as eps needs careful tuning.
            # We'll use a placeholder value.
            logger.debug(f"Using DBSCAN fallback with eps=0.5, min_samples={self.min_cluster_size}")
            clusterer = DBSCAN(eps=0.5, min_samples=self.min_cluster_size, metric="cosine")
            labels = clusterer.fit_predict(ensemble_features)
            logger.info(f"Clustering complete. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters with DBSCAN (fallback).")


        # 4. Group photos based on cluster labels
        clusters: Dict[int, List[PhotoMeta]] = {}
        for photo, label in zip(photos, labels):
            if label == -1:
                # Handle noise points - for now, we can add them to a single "noise" group
                # or treat each as a separate cluster. Let's group them for now.
                label = -1 
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(photo)

        logger.info(f"Grouped photos into {len(clusters)} final clusters (including noise).")
        
        return list(clusters.values())

    async def _extract_all_features(self, photos: List[PhotoMeta]) -> List[Dict[str, np.ndarray]]:
        """
        Extracts all available features for a list of photos.
        """
        logger.debug(f"Extracting features for {len(photos)} photos.")
        feature_list = []
        for photo in photos:
            try:
                image = self.deep_clusterer.convert_image_file(photo.path)
                features_dict = {
                    "clip": self.deep_clusterer.extract_clip_features(image, photo.path),
                    "efficientnet": self.deep_clusterer.extract_efficientnet_features(image, photo.path),
                    "vit": self.deep_clusterer.extract_vit_features(image, photo.path),
                }
                feature_list.append(features_dict)
            except Exception as e:
                logger.error(f"Failed to extract features for {photo.path}: {e}", exc_info=True)
        logger.debug(f"Successfully extracted features for {len(feature_list)} photos.")
        return feature_list

    def _create_ensemble_vectors(self, features_dict_list: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Creates normalized and concatenated feature vectors from a list of feature dictionaries.
        """
        logger.debug(f"Creating ensemble vectors from {len(features_dict_list)} feature sets.")
        ensembled_vectors = []
        for features_dict in features_dict_list:
            # Define which features to use in the ensemble
            # We can make this configurable later
            features_to_ensemble = ['clip', 'efficientnet', 'vit']
            
            vector_parts = []
            for feature_name in features_to_ensemble:
                if feature_name in features_dict and features_dict[feature_name] is not None:
                    # Normalize each feature vector before concatenation
                    part = normalize(features_dict[feature_name].reshape(1, -1), norm='l2').flatten()
                    vector_parts.append(part)
            
            if vector_parts:
                ensembled_vectors.append(np.concatenate(vector_parts))

        return np.array(ensembled_vectors)

