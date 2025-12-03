import logging
from pathlib import Path

from core.deps import get_lock, get_pipeline
from fastapi import APIRouter, Depends, HTTPException
from schema import ClusterGroupResponse, ClusterRequest, ClusterResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/cluster", response_model=ClusterResponse)
async def cluster_images(req: ClusterRequest, 
                         pipeline=Depends(get_pipeline),
                         lock=Depends(get_lock),
                         ) -> ClusterResponse:
    """
    이미지 경로 리스트를 입력받아 클러스터링을 수행하는 엔드포인트.

    - photo_paths: 로컬 파일 시스템 경로들 (예: /Users/you/photos/xxx.jpg)
    - 응답: 각 클러스터의 id, 포함된 사진 경로, 개수, 평균 유사도, quality_score
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="DeepClusterer 가 초기화되지 않았습니다.")

    # 존재하지 않는 파일 체크 (기본적인 검증)
    missing_files = [p for p in req.photo_paths if not Path(p).is_file()]
    logger.info(f"Get Cluster Req {len(req.photo_paths), len(missing_files)}")
    if missing_files:
        raise HTTPException(
            status_code=400,
            detail=f"다음 파일들이 존재하지 않습니다: {missing_files[:5]} "
                   f"{'(외 추가 있음 ...)' if len(missing_files) > 5 else ''}",
        )

    # 요청에서 넘어온 threshold / cache / remove_people 설정을 반영
    # (Lock 안에서 변경 -> 그 클러스터링 작업에만 유효)
    async with lock:
        # 원래 설정 백업
        # orig_threshold = clusterer.similarity_threshold
        # orig_use_cache = clusterer.use_cache
        # orig_remove_people = clusterer.remove_people

        # clusterer.similarity_threshold = req.similarity_threshold
        # clusterer.use_cache = req.use_cache
        # clusterer.remove_people = req.remove_people

        try:
            # 실제 클러스터링 수행
            # cluster() 는 List[List[str]] (클러스터별 경로 리스트)를 반환하지만,
            # 더 자세한 정보는 clusterer.groups 에 들어 있음.
            logger.info(
                f"🚀 Clustering {len(req.photo_paths)} photos "
                f"(threshold={req.similarity_threshold}, "
                f"use_cache={req.use_cache}, remove_people={req.remove_people})"
            )

            # 동기 함수지만, 일단 그냥 호출 (CPU/GPU를 오래 점유하는 동안 이 요청은 블록됨)

            groups = await pipeline.run(req.photo_paths)

            # groups 구조에서 자세한 정보 추출
            clusters: list[ClusterGroupResponse] = []
            total_photos = 0

            for idx, g in enumerate(groups):
                # g 구조:
                # { "id", "photos", "count", "avg_similarity", "quality_score" }
                # total_photos += g["count"]
                # clusters.append(
                #     ClusterGroupResponse(
                #         id=int(g["id"]),
                #         photos=photo_paths,
                #         count=int(g['count']),
                #         avg_similarity=float(g["avg_similarity"]),
                #         quality_score=float(g["quality_score"]),
                #     )
                # )

                photo_paths = [p.path for p in g]
                total_photos += len(g)
                clusters.append(
                    ClusterGroupResponse(
                        id=idx,
                        photos=photo_paths,
                        count=int(len(g)),
                        avg_similarity=1.0,
                        quality_score=1.0,
                    )
                )
            # quality_score 기준으로 이미 정렬되어 있지만, 한 번 더 확실하게 정렬
            clusters.sort(key=lambda c: c.quality_score, reverse=True)

            resp = ClusterResponse(
                clusters=clusters,
                total_photos=total_photos,
                total_clusters=len(clusters),
                similarity_threshold=req.similarity_threshold,
            )
            logger.info(
                f"✅ Clustering done: {resp.total_clusters} clusters, "
                f"{resp.total_photos} photos."
            )
            return resp

        finally:
            # 설정 복원
            pass
            # clusterer.similarity_threshold = orig_threshold
            # clusterer.use_cache = orig_use_cache
            # clusterer.remove_people = orig_remove_people
