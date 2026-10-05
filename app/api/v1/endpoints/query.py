"""POST /api/v1/query — text-only RAG retrieval (no ASR / LLM)."""

from fastapi import APIRouter, Depends, Request

from app.core.security import verify_api_key
from app.schemas.request_schemas import QueryRequest
from app.schemas.response_schemas import QueryResponse
from app.services.rag_service import RAGService

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["rag"],
)
async def query_corpus(payload: QueryRequest, request: Request) -> QueryResponse:
    """Retrieve top-k cosine neighbours from `tech_tutor_corpus`.

    Intended for integration tests and prompt debugging. Does not invoke
    Whisper or Llama 3.
    """
    rag: RAGService = request.app.state.rag
    chunks = await rag.query_context(payload.query_text, top_k=payload.top_k)
    return QueryResponse(
        query_text=payload.query_text,
        top_k=payload.top_k,
        chunks=chunks,
    )
