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


class CLIPDescriptorExtractor:
    """
    전역 임베딩 추출기 (Stage 2).
    - 기본 구현은 CLIP(OpenCLIP ViT-B/32) 기반
    - 프로젝트 상황에 따라 place-recognition 전용 모델로 바꾸기 쉬운 구조
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
        image_size: int = 224,
    ) -> None:
        import open_clip
        import torch

        # device 선택 (mps -> cuda -> cpu)
        if device is not None:
            self.device = torch.device(device)
        else:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        logger.info(f"🔧 Global descriptor device: {self.device}")

        self.model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = preprocess
        self.image_size = image_size

    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        """
        단일 이미지에 대한 L2-normalized descriptor 반환.
        실패하면 None.
        """
        import torch

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"⚠️ Failed to open image {image_path}: {e}")
            return None

        with torch.no_grad():
            image_input = self.preprocess(img).unsqueeze(0).to(self.device)
            features = self.model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
            feat_np: np.ndarray = features.cpu().numpy().flatten().astype(np.float32)
            return feat_np