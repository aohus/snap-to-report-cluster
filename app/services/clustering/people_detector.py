import logging
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from transformers import DetrForObjectDetection, DetrImageProcessor
except Exception:  # pragma: no cover
    DetrForObjectDetection = None  # type: ignore
    DetrImageProcessor = None  # type: ignore

from core.device import DEVICE

logger = logging.getLogger(__name__)



class PeopleDetector:
    def __init__(self, processor, model, device: torch.device):
        self.processor = processor
        self.model = model
        self.device = device

    def __call__(self, images: List[Image.Image], return_tensors: str = "pt"):
        inputs = self.processor(images=images, return_tensors=return_tensors)
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        return inputs

    def post_process_object_detection(self, outputs, **kwargs):
        return self.processor.post_process_object_detection(outputs, **kwargs)

    def mask_people_in_image(self, image: Image.Image) -> Image.Image:
        """
        DETR 기반으로 사람 박스를 찾아 흐리게 처리.
        DETR 모델이 없으면 원본 이미지를 그대로 반환.
        """
        try:
            inputs = self([image], return_tensors="pt")
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([[image.height, image.width]]).to(self.device)
            results = self.post_process_object_detection(
                outputs, target_sizes=target_sizes
            )[0]

            person_boxes: List[List[float]] = []
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                if float(score) < 0.8:
                    continue
                # COCO dataset 에서 person class id == 1
                if int(label) == 1:
                    person_boxes.append([float(x) for x in box.tolist()])

            if not person_boxes:
                return image

            img_np = np.array(image).copy()
            for x_min, y_min, x_max, y_max in person_boxes:
                x_min_i = max(0, int(x_min))
                y_min_i = max(0, int(y_min))
                x_max_i = min(img_np.shape[1], int(x_max))
                y_max_i = min(img_np.shape[0], int(y_max))

                roi = img_np[y_min_i:y_max_i, x_min_i:x_max_i]
                if roi.size == 0:
                    continue
                roi_blur = cv2.GaussianBlur(roi, (21, 21), 0) if cv2 is not None else roi
                img_np[y_min_i:y_max_i, x_min_i:x_max_i] = roi_blur

            return Image.fromarray(img_np)
        except Exception as e:  # pragma: no cover
            logger.warning(f"⚠️ Failed to mask people: {e}")
            return image


def create_people_detector(device: Optional[torch.device] = None) -> Optional[PeopleDetector]:
    """사람 감지 모델 초기화 (없으면 None 반환)."""
    if device is None:
        device = DEVICE

    if DetrForObjectDetection is None or DetrImageProcessor is None:
        logger.warning(
            "⚠️ transformers / DETR 를 찾을 수 없습니다. 사람 제거 기능이 비활성화됩니다."
        )
        return None

    try:
        logger.info("🧍 사람 감지 모델 로딩 중 (DETR ResNet-50)...")
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", use_fast=True)
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        model = model.to(device)
        model.eval()
        return PeopleDetector(processor, model, device)
    except Exception as e:  # pragma: no cover - 환경 의존
        logger.warning(f"⚠️ 사람 감지 모델 로드 실패: {e}")
        return None
