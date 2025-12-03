#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py

맥북 로컬에서 동작하는 "이미지 클러스터링 전용" 서버.
- new_deep_clusterer.DeepClusterer 를 내부에서 사용
- HTTP API 로 이미지 경로 리스트를 전달받아 클러스터링 수행
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from endpoint import router
from fastapi import FastAPI

# new_deep_clusterer.py 가 같은 디렉토리에 있다고 가정
from pipeline import PhotoClusteringPipeline

logger = logging.getLogger("main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Local Image Cluster Server",
    description="맥북 로컬에서 동작하는 이미지 클러스터링 전용 서버 (new_deep_clusterer 기반)",
    version="1.0.0",
)

# DeepClusterer 는 모델 로딩이 무거우므로, 앱 시작 시 1회 초기화해서 재사용
# input_path 는 캐시/결과용 베이스 디렉터리만 의미하므로, 실제 이미지 위치와는 독립적.
BASE_DIR = Path(os.environ.get("IMAGE_CLUSTER_BASE_DIR", ".")).resolve()
CACHE_BASE = BASE_DIR / "cluster_cache"
if not CACHE_BASE.is_dir():
    os.makedirs(CACHE_BASE, exist_ok=True)
    logger.info(f"Created directory: {CACHE_BASE}")

# asyncio Lock 으로 한 번에 하나의 클러스터링 작업만 수행 (모델/상태 공유 보호)
app.state.clusterer_lock = asyncio.Lock()


# 앱 시작 시 초기화될 전역 인스턴스
@app.on_event("startup")
async def startup_event():
    # 여기서는 device 선택을 new_deep_clusterer 내부에 맡김
    # (mps / cuda / cpu 중 가능한 것 자동 선택하는 구조로 만들어 두었음)
    logger.info("🔧 Initializing Pipeline for image clustering server...")
    app.state.pipeline = PhotoClusteringPipeline(CACHE_BASE=CACHE_BASE)
    logger.info("✅ PhotoClusteringPipeline initialized.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down image cluster server...")


app.include_router(router, prefix="/api")


@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": "Welcome to the Photo Clustering API!",
        "docs_url": "/docs",
    }

@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {"status": "ok"}

# ------------------------------------------------------------------------------
# 개발 편의를 위한 로컬 실행 진입점
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # 예: http://127.0.0.1:8001/docs 에서 Swagger UI 확인 가능
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )