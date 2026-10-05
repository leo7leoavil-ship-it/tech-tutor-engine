"""Master APIRouter for version 1."""

from fastapi import APIRouter

from app.api.v1.endpoints import evaluate, health, query, websocket

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(query.router)
api_router.include_router(evaluate.router)
api_router.include_router(websocket.router)
