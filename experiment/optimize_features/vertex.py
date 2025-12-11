import base64
import io
import logging
from concurrent.futures import ThreadPoolExecutor  # I/O 바운드 작업에 적합
from typing import Dict, List, Optional

import numpy as np

# GCP Libraries
from google.cloud import aiplatform
from google.protobuf import struct_pb2
from PIL import Image, ImageFile
from pyproj import Geod

from app.domain.clusterers.base import Clusterer

# 사용자 정의 모듈
from app.domain.storage.factory import get_storage_client
from app.models.photometa import PhotoMeta

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    import hdbscan as HDBSCAN

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- [Vertex AI Client Wrapper] ---

class VertexEmbeddingClient:
    _instance = None
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
        # 멀티모달 임베딩 모델
        self.client = aiplatform.predicition.PredictionServiceClient(
            client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
        )
        self.endpoint = f"projects/{project_id}/locations/{location}/publishers/google/models/multimodalembedding@001"

    @classmethod
    def get_instance(cls):
        # 실제 환경에 맞게 Project ID와 Region 설정 필요
        # 환경 변수에서 가져오도록 수정 권장
        if cls._instance is None:
            # TODO: 프로젝트 ID를 설정 파일이나 환경 변수에서 가져오세요.
            # 예: os.getenv("GOOGLE_CLOUD_PROJECT")
            cls._instance = cls("YOUR_PROJECT_ID", "us-central1") 
        return cls._instance

    def get_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        try:
            encoded_content = base64.b64encode(image_bytes).decode("utf-8")
            instance = struct_pb2.Struct()
            instance.update({"image": {"bytesBase64Encoded": encoded_content}})
            
            instances = [instance]
            # 텍스트 없이 이미지만 보냄 -> 1408차원 벡터 반환
            response = self.client.predict(endpoint=self.endpoint, instances=instances)
            
            embedding = response.predictions[0]['imageEmbedding']
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Vertex AI API Error: {e}")
            return None

def _extract_vertex_feature(path: str) -> Optional[np.ndarray]:
    """
    이미지 다운로드 -> 리사이즈 -> Vertex AI API 호출 -> 임베딩 반환
    """
    try:
        # 1. 이미지 다운로드
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
                # Vertex AI는 전체 이미지를 분석하므로 전체 다운로드 필요
                # 하지만 네트워크 속도를 위해 Resize 후 보낼 것이므로
                # 메모리에 로드 가능한 수준이어야 함. 
                # 원본이 너무 크면(10MB+) Partial로 헤더만 읽기는 불가능.
                # 다행히 Vertex AI는 20MB 제한이 있으므로 대부분 통과.
                img_data = blob.download_as_bytes()
            except Exception:
                return None
        else:
            with open(path, "rb") as f:
                img_data = f.read()

        if not img_data: return None

        # 2. 이미지 전처리 (리사이즈)
        # API 전송 시간 단축 및 비용 절감을 위해 적절히 리사이즈
        with Image.open(io.BytesIO(img_data)) as img:
            img = img.convert("RGB")
            # 긴 변 기준 800px 정도로 리사이즈 (디테일 유지하면서 용량 줄임)
            img.thumbnail((800, 800))
            
            # 다시 바이트로 변환
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=85)
            resized_bytes = output.getvalue()

        # 3. Vertex AI API 호출
        # 여기서 싱글톤 클라이언트 호출
        # (주의: 프로젝트 ID 설정 필요)
        ai_client = VertexEmbeddingClient.get_instance()
        vector = ai_client.get_embedding(resized_bytes)
        
        # 정규화 (Cosine Similarity 계산을 위해 미리 해두면 좋음)
        if vector is not None:
            norm = np.linalg.norm(vector)
            if norm > 0: vector /= norm
            
        return vector

    except Exception as e:
        logger.warning(f"Feature extraction failed for {path}: {e}")
        return None


class HybridCluster(Clusterer):
    def __init__(self):
        self.geod = Geod(ellps="WGS84")
        self.max_gps_tolerance_m = 50.0 
        
        # Vertex AI 임베딩 기준 임계값 (Cosine Distance)
        # 0.0 (동일) ~ 1.0 (다름) ~ 2.0 (정반대)
        # Vertex AI는 의미론적 유사도가 매우 정확하므로 임계값을 타이트하게 잡아도 됨
        self.similarity_strict_thresh = 0.15  # 거의 같은 장소 (계절/공사 유무만 다름)
        self.similarity_loose_thresh = 0.35   # 확실히 다른 장소

    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        # 1. GPS 보정
        self._correct_outliers_by_speed(photos)
        self._adjust_gps_inaccuracy(photos)
        
        valid_photos = [p for p in photos if p.lat is not None and p.lon is not None]
        no_gps_photos = [p for p in photos if p.lat is None or p.lon is None]
        
        if not valid_photos: return [photos]
        
        # 2. Vertex AI 임베딩 추출 (ThreadPool 사용)
        img_paths = [p.path for p in valid_photos]
        logger.info(f"Extracting Vertex AI embeddings for {len(valid_photos)} photos...")
        
        features = []
        if img_paths:
            # API 호출은 I/O 작업이므로 ThreadPool이 훨씬 빠르고 효율적임 (CPU 안 씀)
            # max_workers를 높여서(예: 10~20) 병렬로 API를 쏘면 10초 내 처리 가능
            with ThreadPoolExecutor(max_workers=20) as executor:
                features = list(executor.map(_extract_vertex_feature, img_paths))
        else:
            features = [None] * len(valid_photos)

        # 3. 가중치 거리 행렬 계산
        dist_matrix = self._compute_weighted_distance_matrix(valid_photos, features)
        
        # 4. HDBSCAN 적용
        try:
            clusterer = HDBSCAN(
                min_cluster_size=2,
                min_samples=2,
                metric='precomputed',
                cluster_selection_epsilon=3.0, # 3.0 Weighted Meter
                cluster_selection_method='leaf'
            )
            labels = clusterer.fit_predict(dist_matrix)
        except Exception as e:
            logger.error(f"HDBSCAN error: {e}")
            return [valid_photos]
        
        # 5. 결과 정리
        clusters = self._group_by_labels(valid_photos, labels)
        if no_gps_photos: clusters.append(no_gps_photos)
        return clusters

    def _compute_weighted_distance_matrix(self, photos: List[PhotoMeta], features: List[Optional[np.ndarray]]) -> np.ndarray:
        n = len(photos)
        dist_matrix = np.zeros((n, n))
        coords = np.array([[p.lat, p.lon] for p in photos])
        
        for i in range(n):
            for j in range(i + 1, n):
                _, _, gps_dist = self.geod.inv(coords[i][1], coords[i][0], coords[j][1], coords[j][0])
                
                weight_factor = 1.0 
                
                if features[i] is not None and features[j] is not None:
                    # Cosine Distance (이미 정규화되었으므로 Dot Product만 하면 됨)
                    # 범위: 0(동일) ~ 1(직교) ~ 2(반대)
                    similarity = np.dot(features[i], features[j])
                    struct_dist = 1.0 - similarity
                    
                    # Vertex AI 모델 신뢰도 기반 가중치
                    
                    # Case A: 의미적으로 매우 유사함 (공사 전후, 계절 변화 등은 AI가 '유사'하다고 판단함)
                    if struct_dist < self.similarity_strict_thresh: 
                        # GPS 거리 10%로 축소 -> 강력하게 병합
                        weight_factor = 0.1 
                        
                    # Case B: 의미적으로 다름 (다른 건물, 다른 도로)
                    elif struct_dist > self.similarity_loose_thresh:
                        # GPS 거리 5배 확대 -> 강력하게 분리
                        weight_factor = 5.0 + (struct_dist - 0.35) * 10.0
                    
                    # Case C: 애매함
                    else:
                        weight_factor = 1.0 + (struct_dist - 0.2) * 3.0
                
                else:
                    # 임베딩 실패 시 GPS 의존
                    if gps_dist < 10.0: weight_factor = 1.0
                    else: weight_factor = 2.0

                final_dist = gps_dist * weight_factor
                
                # GPS 물리적 한계 필터
                if gps_dist > self.max_gps_tolerance_m:
                    # AI가 99% 확신하는 경우(0.1)가 아니면 50m 밖은 다른 장소
                    if weight_factor > 0.15: 
                        final_dist = 1000.0

                dist_matrix[i][j] = dist_matrix[j][i] = final_dist
                
        return dist_matrix

    def _correct_outliers_by_speed(self, photos): 
        # (기존 코드 유지)
        timed_photos = [p for p in photos if p.timestamp is not None and p.lat is not None]
        timed_photos.sort(key=lambda x: x.timestamp)
        max_speed_mps = 5.0 
        for i in range(1, len(timed_photos)):
            prev = timed_photos[i-1]
            curr = timed_photos[i]
            dt = curr.timestamp - prev.timestamp
            if dt <= 0: continue
            _, _, dist = self.geod.inv(prev.lon, prev.lat, curr.lon, curr.lat)
            if (dist / dt) > max_speed_mps:
                curr.lat = prev.lat
                curr.lon = prev.lon
                if prev.alt is not None: curr.alt = prev.alt

    def _adjust_gps_inaccuracy(self, photos): 
        # (기존 코드 유지)
        timed_photos = [p for p in photos if p.timestamp is not None]
        timed_photos.sort(key=lambda x: x.timestamp)
        for i in range(len(timed_photos) - 2, -1, -1):
            p1 = timed_photos[i]
            p2 = timed_photos[i+1]
            if 0 <= (p2.timestamp - p1.timestamp) <= 20:
                if p2.lat is not None and p2.lon is not None:
                    p1.lat = p2.lat
                    p1.lon = p2.lon
                    if p2.alt is not None: p1.alt = p2.alt

    def _group_by_labels(self, photos, labels):
        # (기존 코드 유지)
        clusters = {}
        noise = []
        for p, label in zip(photos, labels):
            if label == -1: noise.append(p)
            else: clusters.setdefault(label, []).append(p)
        result = list(clusters.values())
        if noise: 
            for n_photo in noise: result.append([n_photo])
        return result