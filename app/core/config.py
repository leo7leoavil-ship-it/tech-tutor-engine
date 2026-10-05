"""Application settings loaded from environment variables / `.env`."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Tech Tutor MaaS engine.

    All secrets are expected via environment variables (see `.env.example`).
    Vector-store path points at the **pre-built** Data Track ChromaDB;
    this service never recreates the collection.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Tech Tutor Engine"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    api_v1_prefix: str = "/api/v1"
    cors_origins: str = Field(
        default="*",
        description="Comma-separated CORS origins, or '*' to allow all.",
    )

    api_key: SecretStr = Field(
        default=SecretStr("change-me-in-production"),
        description="Bearer token required by REST and WebSocket clients.",
    )

    chroma_persist_dir: str = Field(
        default="./data/technical_corpus_db",
        description="Local path of the pre-built persistent ChromaDB instance.",
    )
    chroma_collection_name: str = "tech_tutor_corpus"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 3

    whisper_model_size: str = "large-v3-turbo"
    whisper_device: str = Field(
        default="cpu",
        description="faster-whisper device: cpu | cuda",
    )
    whisper_compute_type: str = Field(
        default="int8",
        description="CTranslate2 compute type (int8 recommended on CPU Spaces).",
    )
    whisper_beam_size: int = 5
    whisper_language: str | None = Field(
        default="en",
        description="Force ASR language. Set empty/None for auto-detect.",
    )

    llm_provider: Literal["groq", "huggingface", "openai_compatible"] = "groq"
    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Provider API key (Groq / Hugging Face / OpenAI-compatible).",
    )
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Llama 3 family model identifier on the selected provider.",
    )
    llm_base_url: str | None = Field(
        default="https://api.groq.com/openai/v1",
        description="OpenAI-compatible base URL. Groq default; override for HF router.",
    )
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_timeout_seconds: float = 60.0

    skip_heavy_init: bool = Field(
        default=False,
        description="If true, skip Whisper/Chroma/LLM load (unit tests only).",
    )

    ws_max_audio_bytes: int = Field(
        default=15 * 1024 * 1024,
        description="Maximum buffered audio size per WebSocket practice session.",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse CORS origins for FastAPI middleware."""
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
