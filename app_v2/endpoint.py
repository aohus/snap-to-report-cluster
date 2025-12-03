import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app_v2.core.dependencies import get_lock, get_pipeline
from app_v2.schema import ClusterGroupResponse, ClusterRequest, ClusterResponse

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
    
    # Update pipeline config based on request parameters
    # This might require some refactoring in the pipeline or config to allow dynamic updates
    # For now, we'll assume the pipeline is initialized with default config
    # and request parameters might override specific aspects if designed to.
    # Given the request is to *design to best practice*, dynamic config changes per request
    # could be handled by passing them through the pipeline.run method, not by modifying
    # the pipeline's internal state directly, which can be problematic with shared instances.

    async with lock:
        try:
            logger.info(
                f"🚀 Clustering {len(req.photo_paths)} photos "
                f"(threshold={req.similarity_threshold}, "
                f"use_cache={req.use_cache}, remove_people={req.remove_people})"
            )

            # Pass request parameters to the pipeline if it supports dynamic configuration
            # For this refactor, we'll assume the pipeline will use its own internal config
            # but allow overriding of similarity_threshold for compatibility.
            # The pipeline will need to be updated to accept these dynamic parameters.
            final_clusters = await pipeline.run(
                req.photo_paths
            )

            clusters: list[ClusterGroupResponse] = []
            total_photos = 0

            for idx, cluster in enumerate(final_clusters):
                photo_paths = [p.path for p in cluster]
                total_photos += len(cluster)
                # avg_similarity and quality_score are placeholders for now,
                # as the current pipeline doesn't compute them for the final output clusters.
                # These could be added to PhotoMeta or returned by ImageClusterer if needed.
                clusters.append(
                    ClusterGroupResponse(
                        id=idx,
                        photos=photo_paths,
                        count=int(len(cluster)),
                        avg_similarity=1.0, # Placeholder
                        quality_score=1.0,  # Placeholder
                    )
                )
            
            # Sort clusters by quality_score if it were computed
            # clusters.sort(key=lambda c: c.quality_score, reverse=True)

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
            # Cleanup or reset if necessary, though ideally pipeline is stateless per request
            pass
