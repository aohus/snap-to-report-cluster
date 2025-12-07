import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
import logging
from typing import Optional
import numpy as np
from PIL import Image, ImageFile
from config import ClusteringConfig
from core.device import DEVICE
from services.clustering.base import BaseDescriptorExtractor
from services.clustering.people_detector import PeopleDetector
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

# --- Reusing MultiModalNet definition from finetuning/model.py ---
# Ideally this should be in a shared common module, but for now we duplicate 
# or load it. Since we need to load weights, we need the exact class definition.

class MultiModalNet(nn.Module):
    def __init__(self, embedding_dim=128, meta_dim=16, backbone_name='tf_efficientnet_b3_ns'):
        super(MultiModalNet, self).__init__()
        
        # 1. Image Branch
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.img_fc = nn.Linear(self.backbone.num_features, embedding_dim)
        
        # 2. Metadata Branch (Lat, Lon, Timestamp)
        # Input: [lat, lon, timestamp] (These should be normalized/scaled)
        self.meta_fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, meta_dim)
        )
        
        # 3. Fusion
        self.final_fc = nn.Linear(embedding_dim + meta_dim, embedding_dim)
        
    def forward(self, img, meta):
        # Image feature
        x_img = self.backbone(img)
        x_img = self.img_fc(x_img)
        x_img = F.normalize(x_img, p=2, dim=1)
        
        # Meta feature
        x_meta = self.meta_fc(meta)
        x_meta = F.normalize(x_meta, p=2, dim=1)
        
        # Concatenate
        x_combined = torch.cat((x_img, x_meta), dim=1)
        
        # Final embedding
        x_out = self.final_fc(x_combined)
        x_out = F.normalize(x_out, p=2, dim=1)
        
        return x_out


class FinetunedDescriptorExtractor(BaseDescriptorExtractor):
    def __init__(
        self,
        config: ClusteringConfig,
        model_path: str = "finetuning/checkpoints/final_model.pth",
        people_detector: Optional[PeopleDetector] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device if device is not None else DEVICE
        self.config = config.apgem # Reusing APGeM config for image size
        self.people_detector = people_detector
        
        # Load Model
        self.model = MultiModalNet(embedding_dim=128) # Ensure dims match training
        
        # Handle path relative to project root
        resolved_path = Path(model_path)
        if not resolved_path.exists():
             # Try looking in current directory or relative to app
             resolved_path = Path(__file__).resolve().parent.parent.parent.parent / model_path
        
        if resolved_path.exists():
            try:
                state_dict = torch.load(resolved_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded finetuned model from {resolved_path}")
            except Exception as e:
                logger.error(f"Failed to load finetuned model: {e}. Using random weights.")
        else:
            logger.warning(f"Finetuned model not found at {resolved_path}. Using random weights.")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224), # Match training crop
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN,
                    std=IMAGENET_DEFAULT_STD,
                ),
            ]
        )
        
        # Normalization stats (Must match training stats!)
        # These should ideally be saved with the model or config.
        # Using approximations from previous training run logs or dataset stats.
        # Placeholder values - for production, calculate from dataset or save during training.
        self.lat_mean = 37.0
        self.lat_std = 1.0
        self.lon_mean = 127.0
        self.lon_std = 1.0
        self.ts_min = 1600000000.0
        self.ts_range = 100000000.0 


    @torch.no_grad()
    def extract_one(self, image_path: Path, metadata: dict = None) -> Optional[np.ndarray]:
        """
        Extract features using the finetuned multi-modal model.
        Requires metadata (lat, lon, timestamp) to be passed.
        """
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"[Finetuned] Failed to open image {image_path}: {e}")
            return None
        
        if self.people_detector:
            img = self.people_detector.mask_people_in_image(img)

        # Image Tensor
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        # Meta Tensor
        lat = metadata.get('lat') if metadata and metadata.get('lat') is not None else self.lat_mean
        lon = metadata.get('lon') if metadata and metadata.get('lon') is not None else self.lon_mean
        ts = metadata.get('timestamp') if metadata and metadata.get('timestamp') is not None else self.ts_min
        
        norm_lat = (lat - self.lat_mean) / (self.lat_std + 1e-6)
        norm_lon = (lon - self.lon_mean) / (self.lon_std + 1e-6)
        norm_ts = (ts - self.ts_min) / (self.ts_range + 1e-6)
        
        meta_t = torch.tensor([[norm_lat, norm_lon, norm_ts]], dtype=torch.float32).to(self.device)

        # Inference
        feature = self.model(img_t, meta_t)
        
        return feature.cpu().numpy().flatten().astype(np.float32)
