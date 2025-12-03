from fastapi import Depends, Request


def get_pipeline(request: Request):
    return request.app.state.pipeline


def get_lock(request: Request):
    return request.app.state.clusterer_lock