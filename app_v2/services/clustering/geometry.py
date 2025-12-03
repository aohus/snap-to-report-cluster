from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import kornia
import kornia.feature as KF
import torch

from app_v2.core.device import DEVICE
from app_v2.config import ClusteringConfig
from app_v2.services.clustering.base import BaseGeometryMatcher
from app_v2.services.clustering.segmenter import SemanticSegmenter

logger = logging.getLogger(__name__)


class SIFTMatcher(BaseGeometryMatcher):
    def __init__(self, config: ClusteringConfig) -> None:
        self.config = config.sift
        self.enabled = False
        self.detector = None

        try:
            self.detector = cv2.SIFT_create(nfeatures=self.config.max_features)
            self.enabled = True
        except Exception:
            logger.warning(
                "SIFT 생성 실패 (opencv-contrib-python 필요). "
                "SIFT 매칭 비활성화."
            )

    def _load_gray(self, path: Path) -> Optional[np.ndarray]:
        if not self.enabled or self.detector is None:
            return None
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Failed to read image for SIFT: {path}")
            return None
        return img

    def geo_score(self, path1: Path, path2: Path) -> float:
        if not self.enabled or self.detector is None:
            return 1.0  # Fallback: always pass if SIFT is disabled

        img1 = self._load_gray(path1)
        img2 = self._load_gray(path2)
        if img1 is None or img2 is None:
            return 0.0

        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < self.config.ratio_thresh * n.distance:
                good.append(m)

        if len(good) < self.config.min_good_matches:
            return 0.0

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, self.config.ransac_reproj_thresh
        )
        if H is None or mask is None:
            return 0.0

        inliers = int(np.sum(mask))
        total = len(good)
        if total == 0:
            return 0.0
        score = float(inliers) / total
        return max(0.0, min(1.0, score))


class LoFTRMatcher(BaseGeometryMatcher):
    def __init__(self, config: ClusteringConfig, segmenter: Optional[SemanticSegmenter] = None) -> None:
        self.config = config.loftr
        self.device = DEVICE
        self.loftr = KF.LoFTR(pretrained=self.config.pretrained).to(self.device)
        self.segmenter = segmenter

    def _load_gray_tensor(self, path: Path, apply_mask: bool) -> Optional[torch.Tensor]:
        try:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Failed to read image for LoFTR: {path}")
                return None
            
            # Resize image to LoFTR's expected input size
            img = cv2.resize(img, (640, 480))
            tensor = kornia.image_to_tensor(img, keepdim=False).float() / 255.0

            if tensor.dim() == 3: # Should be (1,H,W)
                tensor = tensor.unsqueeze(0) # Becomes (1,1,H,W)
            
            if apply_mask and self.segmenter:
                mask = self.segmenter.create_mask(path)
                mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)
                bool_mask = torch.from_numpy(mask) == 0 # 0 means masked out
                tensor.squeeze()[bool_mask] = 0.0 # Set masked regions to black
            
            return tensor.to(self.device)
        except Exception as e:
            logger.warning(f"Failed to load/process {path} for LoFTR: {e}")
            return None

    @torch.no_grad()
    def geo_score(self, path1: Path, path2: Path) -> float:
        # Use semantic mask only if configured and segmenter is available
        apply_mask = ClusteringConfig().use_semantic_mask_for_loftr and self.segmenter is not None
        
        t1 = self._load_gray_tensor(path1, apply_mask=apply_mask)
        t2 = self._load_gray_tensor(path2, apply_mask=apply_mask)
        
        if t1 is None or t2 is None:
            return 0.0

        input_dict = {"image0": t1, "image1": t2}
        correspondences = self.loftr(input_dict)

        confidences = correspondences['confidence'].cpu().numpy()
        num_confident_matches = np.sum(confidences > self.config.confidence_threshold)
        
        # Heuristic normalization
        score = min(num_confident_matches / 100.0, 1.0)
        return score
