from typing import Optional
from pydantic import BaseModel, Field, validator, HttpUrl


class ClusterRequest(BaseModel):
    photo_paths: list[str] = Field(
        ..., description="클러스터링할 이미지의 절대 경로(or 신뢰 가능한 로컬 경로) 리스트"
    )
    similarity_threshold: float = Field(
        0.3, ge=0.0, le=1.0, description="전역 임베딩 코사인 유사도 임계값 (0~1)"
    )
    use_cache: bool = Field(
        True, description="특징 벡터 캐시 사용 여부"
    )
    remove_people: bool = Field(
        True, description="사람 영역을 마스킹할지 여부 (DETR 사용)"
    )
    webhook_url: Optional[HttpUrl] = Field(
        None, description="작업 완료 후 결과를 수신할 Webhook URL (POST 요청)"
    )
    request_id: Optional[str] = Field(
        None, description="클라이언트 측 트래킹 ID"
    )

    @validator("photo_paths")
    def validate_paths(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("photo_paths 는 비어 있을 수 없습니다.")
        return v


class ClusterGroupResponse(BaseModel):
    id: int
    photos: list[str]
    count: int
    avg_similarity: float
    quality_score: float


class ClusterResponse(BaseModel):
    clusters: list[ClusterGroupResponse]
    total_photos: int
    total_clusters: int
    similarity_threshold: float


class ClusterTaskResponse(BaseModel):
    task_id: str
    status: str = "processing"
    message: str = "Clustering task accepted."
    request_id: Optional[str] = None
