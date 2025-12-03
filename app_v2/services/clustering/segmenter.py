from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor

from app_v2.core.device import DEVICE
from app_v2.config import ClusteringConfig

logger = logging.getLogger(__name__)

class SemanticSegmenter:
    """
    Detects specific classes in an image using a deep learning model
    and generates a mask. The generated masks are cached for reuse.
    """
    def __init__(self, config: ClusteringConfig):
        self.device = DEVICE
        self.config = config.semantic_segmenter

        self.model = torch.hub.load('pytorch/vision:v0.13.0', self.config.model_name, pretrained=True)
        self.model.to(self.device).eval()
        
        self.coco_class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/N', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        self.mask_class_indices = {self.coco_class_names.index(name) for name in self.config.classes_to_mask if name in self.coco_class_names}
        logger.info(f"[SemanticSegmenter] Masking classes: {[self.coco_class_names[i] for i in self.mask_class_indices]}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._mask_cache: Dict[Path, np.ndarray] = {} # Consider a more persistent cache for production

    @torch.no_grad()
    def create_mask(self, image_path: Path) -> np.ndarray:
        if image_path in self._mask_cache:
            return self._mask_cache[image_path]

        try:
            input_image = Image.open(image_path).convert("RGB")
            original_size = input_image.size
            input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)

            output = self.model(input_tensor)['out'][0]
            output_predictions = output.argmax(0).byte().cpu()
            
            output_pil = to_pil_image(output_predictions).resize(original_size, Image.NEAREST)
            output_predictions = to_tensor(output_pil).squeeze(0).byte()

            mask = torch.ones_like(output_predictions, dtype=torch.uint8)
            for class_idx in self.mask_class_indices:
                mask[output_predictions == class_idx] = 0

            mask_np = mask.numpy()
            self._mask_cache[image_path] = mask_np
            return mask_np
        except Exception as e:
            logger.warning(f"Failed to create semantic mask for {image_path}: {e}")
            img = Image.open(image_path)
            # Return an all-ones mask if segmentation fails, meaning no part is masked out
            return np.ones((img.height, img.width), dtype=np.uint8)
