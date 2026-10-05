"""FastAPI entry point: lifespan, CORS, and versioned routers."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.v1.endpoints import health as health_endpoint
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.services.asr_service import ASRService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load Whisper, bind ChromaDB, and construct the Llama 3 client once."""
    settings: Settings = app.state.settings
    _configure_logging(settings.debug)

    rag = RAGService(settings)
    asr = ASRService(settings)
    llm = LLMService(settings)
    if settings.skip_heavy_init:
        logger.warning("SKIP_HEAVY_INIT=true — Chroma/Whisper/LLM weights are not loaded.")
    else:
        rag.connect()
        asr.load()
        llm.load()

    app.state.rag = rag
    app.state.asr = asr
    app.state.llm = llm
    logger.info("Tech Tutor Engine ready (v%s)", __version__)
    yield
    logger.info("Tech Tutor Engine shutting down")


def create_app() -> FastAPI:
    """Application factory used by Uvicorn and tests."""
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Decoupled Model-as-a-Service for technical-English evaluation: "
            "Whisper ASR, ChromaDB RAG (`tech_tutor_corpus`), and Llama 3 structured feedback."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    application.state.settings = settings
    allow_credentials = settings.cors_origin_list != ["*"]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(health_endpoint.router)
    application.include_router(api_router, prefix=settings.api_v1_prefix)
    return application


app = create_app()
