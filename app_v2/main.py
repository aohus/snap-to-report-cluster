#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py

FastAPI server for image clustering using the refactored app_v2 components.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from app_v2.config import ClusteringConfig
from app_v2.services.pipeline import PhotoClusteringPipeline
from app_v2.endpoint import router as clustering_router

logger = logging.getLogger("app_v2_main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Local Image Cluster Server v2",
    description="Local server for image clustering (app_v2 refactored version)",
    version="2.0.0",
)

# Base directory for cache and other assets
BASE_DIR = Path(os.environ.get("IMAGE_CLUSTER_BASE_DIR", ".")).resolve()
# Example cache directory, should be handled by individual components if needed
# CACHE_BASE = BASE_DIR / "cluster_cache" 
# if not CACHE_BASE.is_dir():
#     os.makedirs(CACHE_BASE, exist_ok=True)
#     logger.info(f"Created directory: {CACHE_BASE}")

# asyncio Lock to ensure only one clustering task runs at a time
app.state.clusterer_lock = asyncio.Lock()


@app.on_event("startup")
async def startup_event():
    logger.info("🔧 Initializing PhotoClusteringPipeline for app_v2 server...")
    # Initialize the ClusteringConfig
    clustering_config = ClusteringConfig()
    app.state.pipeline = PhotoClusteringPipeline(config=clustering_config)
    logger.info("✅ PhotoClusteringPipeline initialized.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down app_v2 image cluster server...")


app.include_router(clustering_router, prefix="/api")


@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": "Welcome to the Photo Clustering API v2!",
        "docs_url": "/docs",
    }

@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app_v2.main:app", # Point to the app in app_v2
        host="0.0.0.0",
        port=8002, # Using a different port for v2
        reload=True,
    )
