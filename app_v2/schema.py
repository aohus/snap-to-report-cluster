from pydantic import BaseModel, Field, validator


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