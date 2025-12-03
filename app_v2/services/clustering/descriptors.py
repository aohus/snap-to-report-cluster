from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

from app_v2.core.device import DEVICE
from app_v2.config import ClusteringConfig
from app_v2.services.clustering.base import BaseDescriptorExtractor
from app_v2.services.clustering.people_detector import PeopleDetector # New import

logger = logging.getLogger(__name__)


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps)
        x = x ** self.p
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))
        x = x ** (1.0 / self.p)
        return x


class APGeMDescriptorExtractor(BaseDescriptorExtractor):
    def __init__(
        self,
        config: ClusteringConfig,
        people_detector: Optional[PeopleDetector] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device if device is not None else DEVICE
        self.config = config.apgem
        self.people_detector = people_detector

        self.backbone = timm.create_model(
            self.config.model_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        self.backbone.to(self.device)
        self.backbone.eval()

        self.feat_dim = self.backbone.num_features
        logger.info(f"[APGeM] backbone={self.config.model_name}, feat_dim={self.feat_dim}, device={self.device}")

        self.pool = GeM(p=3.0).to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize(int(self.config.image_size * 1.14)),
                transforms.CenterCrop(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN,
                    std=IMAGENET_DEFAULT_STD,
                ),
            ]
        )



    @torch.no_grad()
    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"[APGeM] Failed to open image {image_path}: {e}")
            return None
        
        if self.people_detector:
            img = self.people_detector.mask_people_in_image(img)

        x = self.transform(img).unsqueeze(0).to(self.device)
        feat_map = self.backbone(x)

        if feat_map.ndim == 2:
            desc = feat_map
        else:
            pooled = self.pool(feat_map)
            desc = pooled.flatten(1)

        desc = F.normalize(desc, p=2, dim=1)
        return (
            desc.squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )


class CLIPDescriptorExtractor(BaseDescriptorExtractor):
    def __init__(
        self,
        config: ClusteringConfig,
        people_detector: Optional[PeopleDetector] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device if device is not None else DEVICE
        self.config = config.clip
        self.people_detector = people_detector

        import open_clip # Lazy import
        self.model, _, preprocess = open_clip.create_model_and_transforms(
            self.config.model_name, pretrained=self.config.pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = preprocess
        logger.info(f"[CLIP] model={self.config.model_name}, pretrained={self.config.pretrained}, device={self.device}")


    @torch.no_grad()
    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"[CLIP] Failed to open image {image_path}: {e}")
            return None

        if self.people_detector:
            img = self.people_detector.mask_people_in_image(img)

        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        features = self.model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)
        feat_np = features.cpu().numpy().flatten().astype(np.float32)
        return feat_np


class CombinedAPGeMCLIPExtractor(BaseDescriptorExtractor):
    def __init__(
        self,
        config: ClusteringConfig,
        people_detector: Optional[PeopleDetector] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device if device is not None else DEVICE
        self.descriptor_config = config.descriptor
        
        self.apgem = APGeMDescriptorExtractor(config, people_detector=people_detector, device=self.device)
        self.clip = CLIPDescriptorExtractor(config, people_detector=people_detector, device=self.device)
        self.w_apgem = self.descriptor_config.w_apgem
        self.w_clip = self.descriptor_config.w_clip
        # The experiment code had l2_normalize_final, but for combined features, it's generally good practice to always normalize.
        self.l2_normalize_final = True 

    @torch.no_grad()
    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        f_ap = self.apgem.extract_one(image_path)
        f_cl = self.clip.extract_one(image_path)

        if f_ap is None and f_cl is None:
            return None

        parts: List[np.ndarray] = []

        if f_ap is not None:
            fa = f_ap.astype(np.float32)
            fa = fa / (np.linalg.norm(fa) + 1e-8)
            fa = fa * self.w_apgem
            parts.append(fa)

        if f_cl is not None:
            fc = f_cl.astype(np.float32)
            fc = fc / (np.linalg.norm(fc) + 1e-8)
            fc = fc * self.w_clip
            parts.append(fc)

        if not parts:
            return None

        combined = np.concatenate(parts).astype(np.float32)

        if self.l2_normalize_final:
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm

        return combined