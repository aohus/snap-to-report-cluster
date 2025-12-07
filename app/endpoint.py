import asyncio
import logging
import traceback
import uuid
from pathlib import Path

from core.dependencies import get_lock, get_pipeline
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from schema import (
    ClusterGroupResponse,
    ClusterPhoto, # Import added
    ClusterRequest,
    ClusterResponse,
    ClusterTaskResponse,
)
from services.callback_sender import CallbackSender
from services.pipeline import PhotoClusteringPipeline

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
        missing_files = [p for p in req.photo_paths if not p.startswith("http") and not Path(p).is_file()]
        if missing_files:
            raise ValueError(f"Files not found: {missing_files[:3]}...")

        # 2. Run Pipeline (Protected by Lock)
        async with lock:
            # TODO: Pass req configuration (thresholds, etc.) to pipeline.run if dynamic config is supported
            final_clusters = await pipeline.run(req.photo_paths)

        # 3. Format Response
        clusters = []
        total_photos = 0
        
        # Separate single items as noise
        valid_clusters = []
        noise_photos = []
        
        for cluster in final_clusters:
            if len(cluster) == 1:
                noise_photos.extend(cluster)
            else:
                valid_clusters.append(cluster)
        
        # Process valid clusters
        for idx, cluster in enumerate(valid_clusters):
            photo_paths = [p.path for p in cluster]
            photo_details = [
                ClusterPhoto(
                    path=p.path,
                    timestamp=p.timestamp,
                    lat=p.lat,
                    lon=p.lon
                ) for p in cluster
            ]
            total_photos += len(cluster)
            clusters.append(
                ClusterGroupResponse(
                    id=idx,
                    photos=photo_paths,
                    photo_details=photo_details,
                    count=int(len(cluster)),
                    avg_similarity=1.0, # Placeholder as per current logic
                    quality_score=1.0,  # Placeholder
                )
            )
        
        # Add noise cluster if any
        if noise_photos:
            noise_paths = [p.path for p in noise_photos]
            noise_details = [
                ClusterPhoto(
                    path=p.path,
                    timestamp=p.timestamp,
                    lat=p.lat,
                    lon=p.lon
                ) for p in noise_photos
            ]
            total_photos += len(noise_photos)
            clusters.append(
                ClusterGroupResponse(
                    id=-1, # Special ID for noise
                    photos=noise_paths,
                    photo_details=noise_details,
                    count=len(noise_photos),
                    avg_similarity=0.0,
                    quality_score=0.0
                )
            )
        
        # Sort clusters by size (descending), keeping noise at the end usually or handled by ID
        # valid clusters sorted by size
        # clusters.sort(key=lambda c: c.count, reverse=True) # This might mix -1.
        # Better to keep valid ones sorted, append noise.
        
        # (Already appended noise at the end)

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
    logger.info(f"📥 [Task] Request accepted. Request ID: {req.request_id}, Webhook: {req.webhook_url}")
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