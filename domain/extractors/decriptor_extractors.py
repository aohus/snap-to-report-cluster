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

logger = logging.getLogger(__name__)


class BaseDescriptorExtractor:
    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        raise NotImplementedError


# -------------------------------------------------------------------
# GeM + APGeMDescriptorExtractor
# -------------------------------------------------------------------

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
        return x  # (B, C, 1, 1)


class APGeMDescriptorExtractor(BaseDescriptorExtractor):
    """
    EfficientNet + GeM 기반 AP-GeM 계열 글로벌 디스크립터
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnet_b3_ns",
        image_size: int = 320,
        device: Optional[str] = None,
    ) -> None:
        if device is not None:
            self.device = torch.device(device)
        else:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        logger.info(f"[APGeM] using device = {self.device}")

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        self.backbone.to(self.device)
        self.backbone.eval()

        self.feat_dim = self.backbone.num_features
        logger.info(f"[APGeM] backbone={model_name}, feat_dim={self.feat_dim}")

        self.pool = GeM(p=3.0).to(self.device)

        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
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


# -------------------------------------------------------------------
# CLIPDescriptorExtractor
# -------------------------------------------------------------------

class CLIPDescriptorExtractor(BaseDescriptorExtractor):
    """
    OpenCLIP 기반 글로벌 디스크립터
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
    ) -> None:
        if device is not None:
            self.device = torch.device(device)
        else:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        logger.info(f"[CLIP] using device = {self.device}")

        self.model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = preprocess

    @torch.no_grad()
    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"[CLIP] Failed to open image {image_path}: {e}")
            return None

        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        features = self.model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)
        feat_np = features.cpu().numpy().flatten().astype(np.float32)
        return feat_np


# -------------------------------------------------------------------
# Combined APGeM + CLIP Extractor
# -------------------------------------------------------------------

class CombinedAPGeMCLIPExtractor(BaseDescriptorExtractor):
    """
    APGeM + CLIP 조합 디스크립터
    - 각 디스크립터를 L2 normalize한 뒤, 가중치 곱해 concat
    - 마지막에 한 번 더 L2 normalize (선택)
    """

    def __init__(
        self,
        apgem: Optional[APGeMDescriptorExtractor] = None,
        clip: Optional[CLIPDescriptorExtractor] = None,
        w_apgem: float = 0.7,
        w_clip: float = 0.3,
        l2_normalize_final: bool = True,
    ) -> None:
        # 같은 device를 쓰는 게 좋음 (여기서는 각각 내부에서 device 결정)
        self.apgem = apgem or APGeMDescriptorExtractor()
        self.clip = clip or CLIPDescriptorExtractor()
        self.w_apgem = float(w_apgem)
        self.w_clip = float(w_clip)
        self.l2_normalize_final = bool(l2_normalize_final)

    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        f_ap = self.apgem.extract_one(image_path)
        f_cl = self.clip.extract_one(image_path)

        if f_ap is None and f_cl is None:
            return None

        parts: list[np.ndarray] = []

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
