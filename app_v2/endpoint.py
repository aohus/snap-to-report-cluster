import asyncio
import logging
import uuid
import traceback
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from app_v2.core.dependencies import get_lock, get_pipeline
from app_v2.schema import ClusterRequest, ClusterResponse, ClusterTaskResponse, ClusterGroupResponse
from app_v2.services.callback_sender import CallbackSender
from app_v2.services.pipeline import PhotoClusteringPipeline

logger = logging.getLogger(__name__)
router = APIRouter()
callback_sender = CallbackSender()

async def process_clustering_task(
    task_id: str,
    req: ClusterRequest,
    pipeline: PhotoClusteringPipeline,
    lock: asyncio.Lock
):
    logger.info(f"🏁 [Task {task_id}] Background processing started. Photos: {len(req.photo_paths)}")
    
    try:
        # 1. Validate files exist (Fail fast)
        missing_files = [p for p in req.photo_paths if not Path(p).is_file()]
        if missing_files:
            raise ValueError(f"Files not found: {missing_files[:3]}...")

        # 2. Run Pipeline (Protected by Lock)
        async with lock:
            # TODO: Pass req configuration (thresholds, etc.) to pipeline.run if dynamic config is supported
            final_clusters = await pipeline.run(req.photo_paths)

        # 3. Format Response
        clusters = []
        total_photos = 0
        for idx, cluster in enumerate(final_clusters):
            photo_paths = [p.path for p in cluster]
            total_photos += len(cluster)
            clusters.append(
                ClusterGroupResponse(
                    id=idx,
                    photos=photo_paths,
                    count=int(len(cluster)),
                    avg_similarity=1.0, # Placeholder as per current logic
                    quality_score=1.0,  # Placeholder
                )
            )
        
        # Sort clusters by size (descending)
        clusters.sort(key=lambda c: c.count, reverse=True)

        response_payload = ClusterResponse(
            clusters=clusters,
            total_photos=total_photos,
            total_clusters=len(clusters),
            similarity_threshold=req.similarity_threshold
        ).dict()
        
        # Add status context
        full_payload = {
            "task_id": task_id,
            "request_id": req.request_id,
            "status": "completed",
            "result": response_payload
        }

        logger.info(f"✅ [Task {task_id}] Processing completed. Clusters: {len(clusters)}")

        # 4. Send Callback
        if req.webhook_url:
            await callback_sender.send_result(str(req.webhook_url), full_payload, task_id)
        else:
            logger.warning(f"⚠️ [Task {task_id}] No webhook_url provided. Result is lost (not persisted).")

    except Exception as e:
        logger.error(f"💥 [Task {task_id}] Processing failed: {e}")
        logger.error(traceback.format_exc())
        if req.webhook_url:
            await callback_sender.send_error(str(req.webhook_url), str(e), task_id, req.request_id)


@router.post("/cluster", response_model=ClusterTaskResponse, status_code=202)
async def submit_cluster_task(
    req: ClusterRequest,
    background_tasks: BackgroundTasks,
    pipeline: PhotoClusteringPipeline = Depends(get_pipeline),
    lock: asyncio.Lock = Depends(get_lock),
):
    """
    Submit an asynchronous image clustering task.
    
    - Returns `202 Accepted` immediately with a `task_id`.
    - Processes the clustering in the background.
    - Sends the result to the provided `webhook_url` (POST) upon completion.
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="DeepClusterer service not initialized.")

    # Create a unique task ID
    task_id = str(uuid.uuid4())
    
    # Add the processing task to background tasks
    background_tasks.add_task(
        process_clustering_task,
        task_id,
        req,
        pipeline,
        lock
    )

    logger.info(f"📥 [Task {task_id}] Request accepted. Request ID: {req.request_id}, Webhook: {req.webhook_url}")

    return ClusterTaskResponse(
        task_id=task_id,
        request_id=req.request_id,
        status="processing",
        message="Clustering task started in background."
    )