import io
import logging
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np

# from app.services.vertex_client import VertexEmbeddingClient 
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
from pyproj import Geod
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from app.domain.clusterers.base import Clusterer
from app.domain.storage.factory import get_storage_client
from app.models.photometa import PhotoMeta

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class FeatureExtractor:
    _instance = None
    _model = None
    _preprocess = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            # CPU 모드로 경량 모델 로드
            # pretrained=True: 이미지넷 데이터로 학습된 가중치 사용 (사물의 특징을 잘 앎)
            weights = MobileNet_V3_Small_Weights.DEFAULT
            cls._model = mobilenet_v3_small(weights=weights)
            cls._model.eval() # 평가 모드 (학습 X)
            
            # 마지막 분류 레이어(Classifier) 제거 -> 특징 벡터(Embedding)만 추출
            # MobileNetV3 Small의 마지막 features 출력은 576차원
            cls._model.classifier = torch.nn.Identity()
            
            cls._preprocess = weights.transforms()
        return cls._model, cls._preprocess


def _extract_mobilenet_feature(path: str) -> Optional[np.ndarray]:
    """
    MobileNetV3를 사용하여 이미지의 의미론적 특징(Semantic Feature) 추출
    """
    try:
        # 1. 이미지 다운로드 (기존 동일)
        storage = get_storage_client()
        img_data = None
        
        if "storage.googleapis.com" in path or path.startswith("gs://"):
            try:
                if path.startswith("gs://"):
                    blob_name = path.replace("gs://", "").split("/", 1)[1]
                else:
                    blob_name = path.split("storage.googleapis.com/")[1].split("/", 1)[1]
                bucket = storage.bucket
                blob = bucket.blob(blob_name)
                # 딥러닝 모델은 전체 이미지가 필요할 수 있으나, 
                # MobileNet은 224x224로 리사이즈 하므로 100KB 정도면 충분
                img_data = blob.download_as_bytes(start=0, end=100 * 1024)
            except: return None
        else:
            with open(path, "rb") as f: img_data = f.read()

        if not img_data: return None

        # 2. 전처리
        with Image.open(io.BytesIO(img_data)) as img:
            img = img.convert("RGB") # PyTorch 모델은 RGB 3채널 필요
            
            # 모델 로드 (싱글톤 패턴 활용)
            model, preprocess = FeatureExtractor.get_model()
            
            # 이미지 텐서 변환 및 정규화
            input_tensor = preprocess(img).unsqueeze(0) # Batch 차원 추가
            
            # 3. 추론 (Inference) - CPU
            with torch.no_grad():
                feature_vector = model(input_tensor)
            
            # (1, 576) -> (576,) numpy array
            return feature_vector.squeeze().numpy()

    except Exception as e:
        logger.error(f"MobileNet extraction failed: {e}")
        return None
    


def prepare_dataset():
    """
    정답셋 1000장의 임베딩을 미리 추출하여 저장합니다.
    """
    # 1. 정답셋 로딩 (예시: CSV나 DB에서 로드한다고 가정)
    # photos 리스트에는 각 사진의 정답 라벨(label_id)이 있어야 합니다.
    # photos: List[PhotoMeta] = load_ground_truth_photos()
    
    # 2. 임베딩 추출 (Vertex AI)
    # 비용 절감을 위해 여기서 한 번만 API를 호출합니다.
    # vertex_client = VertexEmbeddingClient.get_instance()
    
    features = []
    valid_photos = []
    
    print("Extracting embeddings for optimization...")
    for p in photos:
        # 로컬 이미지 로드 및 리사이즈 로직 필요 (이전 코드 참조)
        # img_bytes = load_and_resize(p.path)
        # vector = vertex_client.get_embedding(img_bytes)
        vector = _extract_mobilenet_feature(p.path)
        
        # 테스트용 더미 데이터 (실제 실행 시엔 위 로직 사용)
        vector = np.random.rand(1408).astype(np.float32) 
        
        if vector is not None:
            features.append(vector)
            valid_photos.append(p)
            
    # 3. 데이터 저장 (Pickle)
    # 나중에 Optuna가 이 파일을 계속 재사용합니다.
    with open("dataset_cache.pkl", "wb") as f:
        pickle.dump({
            "photos": valid_photos,
            "features": features
        }, f)
    
    print(f"Saved {len(features)} features to dataset_cache.pkl")

# 실행
# prepare_dataset()