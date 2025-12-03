from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class GPSConfig:
    eps_m: float = 18.0
    min_samples: int = 3

@dataclass
class APGeMConfig:
    model_name: str = "tf_efficientnet_b3_ns"
    image_size: int = 320

@dataclass
class CLIPConfig:
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    
@dataclass
class DescriptorConfig:
    w_apgem: float = 0.7
    w_clip: float = 0.3
    similarity_threshold: float = 0.8
    knn_k: int = 10

@dataclass
class SIFTConfig:
    max_features: int = 1500
    ratio_thresh: float = 0.75
    ransac_reproj_thresh: float = 5.0
    min_good_matches: int = 10

@dataclass
class LoFTRConfig:
    confidence_threshold: float = 0.8
    pretrained: str = "outdoor"
    
@dataclass
class SemanticSegmenterConfig:
    model_name: str = 'deeplabv3_resnet101'
    classes_to_mask: List[str] = field(default_factory=lambda: [
        'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'boat',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'potted plant', 'tv',
        'laptop', 'chair', 'couch', 'dining table'
    ])

@dataclass
class DeepClusterConfig:
    geo_threshold: float = 0.25 # Default from img_clusterer
    min_cluster_size: int = 2
    
@dataclass
class ClusteringConfig:
    gps: GPSConfig = field(default_factory=GPSConfig)
    apgem: APGeMConfig = field(default_factory=APGeMConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    descriptor: DescriptorConfig = field(default_factory=DescriptorConfig)
    sift: SIFTConfig = field(default_factory=SIFTConfig)
    loftr: LoFTRConfig = field(default_factory=LoFTRConfig)
    semantic_segmenter: SemanticSegmenterConfig = field(default_factory=SemanticSegmenterConfig)
    deep_cluster: DeepClusterConfig = field(default_factory=DeepClusterConfig)

    # General settings
    use_semantic_mask_for_loftr: bool = True # This can be set based on request or config
    cache_dir: str = ".cache"
    use_cache: bool = True
    remove_people: bool = True # For people detector (used in some extractors)

