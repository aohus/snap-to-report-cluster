from fastapi import Depends, Request

from app_v2.services.pipeline import PhotoClusteringPipeline

def get_pipeline(request: Request) -> PhotoClusteringPipeline:
    return request.app.state.pipeline

def get_lock(request: Request):
    return request.app.state.clusterer_lock
