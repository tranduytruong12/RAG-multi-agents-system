# FILE: customer_support_rag/config/settings.py
"""Application settings loaded from environment variables via Pydantic BaseSettings.

All configuration for every phase (ingestion, retrieval, agents, API) lives here.
Values are read from the .env file at startup. Defaults are provided for all
optional settings so the system can boot in development without a full .env.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object for the Multi-Agent RAG system.

    Loaded once at application startup. Use the module-level `settings` singleton
    rather than instantiating this class directly.

    Example:
        from config.settings import settings
        print(settings.llm_model_name)  # "gpt-4o"
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── LLM Provider ─────────────────────────────────────────────────────────
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. Required when llm_provider='openai'.",
    )
    llm_model_name: str = Field(
        default="gpt-4o",
        description="LLM model identifier. E.g. 'gpt-4o' or 'llama3' for Ollama.",
    )
    llm_provider: str = Field(
        default="openai",
        description="Active LLM provider: 'openai' or 'ollama'. Switch to 'ollama' for local inference.",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature. 0.0 = deterministic (recommended for agents).",
    )

    # ── Ollama (Future Expansion) ─────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama API server. Used when llm_provider='ollama'.",
    )
    ollama_model_name: str = Field(
        default="llama3",
        description="Ollama model name to use when llm_provider='ollama'.",
    )

    # ── Cohere (Reranker) ─────────────────────────────────────────────────────
    cohere_api_key: str = Field(
        default="",
        description="Cohere API key. Required for CohereReranker in retrieval/reranker.py.",
    )
    reranker_top_n: int = Field(
        default=5,
        gt=0,
        description="Number of top results to return after Cohere reranking.",
    )

    # —— LangSmith / LangChain Observability ——————————————————————————————————————
    langsmith_api_key: str = Field(
        default="",
        description=(
            "Preferred LangSmith API key (lsv2_...). If set, this takes precedence "
            "for LangSmith client authentication."
        ),
    )
    langchain_api_key: str = Field(
        default="",
        description=(
            "Legacy alias for LangSmith API key. Kept for compatibility with older "
            "LANGCHAIN_* env naming."
        ),
    )
    langsmith_endpoint: str = Field(
        default="",
        description="Optional LangSmith API base URL override.",
    )
    langchain_endpoint: str = Field(
        default="",
        description="Legacy alias for LangSmith endpoint override.",
    )
    langchain_project: str = Field(
        default="rag-eval",
        description="Default LangSmith project name for traces/evaluation runs.",
    )
    langchain_tracing_v2: bool = Field(
        default=True,
        description="Enable LangSmith tracing v2.",
    )

    # ── Embeddings ────────────────────────────────────────────────────────────
    embed_model_name: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name.",
    )
    embed_dimensions: int = Field(
        default=1536,
        gt=0,
        description="Dimensionality of the embedding vectors.",
    )

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_host: str = Field(
        default="localhost",
        description="ChromaDB server host.",
    )
    chroma_port: int = Field(
        default=8001,
        gt=0,
        lt=65536,
        description="ChromaDB server port.",
    )
    chroma_collection_name: str = Field(
        default="support_docs",
        description="Name of the ChromaDB collection for support documents.",
    )

    # ── Ingestion ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(
        default=512,
        gt=0,
        description="Target token count per document chunk.",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Token overlap between consecutive chunks to preserve context.",
    )

    # ── Agent Configuration ───────────────────────────────────────────────────
    agent_max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum QA retry loops before the orchestrator escalates to a human.",
    )

    # ── Conversation Memory / Query Rewriting ─────────────────────────────────
    memory_window_size: int = Field(
        default=4,
        ge=1,
        le=10,
        description=(
            "Number of recent conversation turns to keep per session for contextual "
            "query rewriting. Sourced from MEMORY_WINDOW_SIZE env var."
        ),
    )
    enable_query_rewriting: bool = Field(
        default=True,
        description=(
            "Enable contextual query rewriting for follow-up queries. "
            "Set ENABLE_QUERY_REWRITING=false to disable. "
            "When disabled, queries are passed to the pipeline unchanged."
        ),
    )

    # ── FastAPI ───────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="API server bind host.")
    api_port: int = Field(default=8000, gt=0, description="API server bind port. Default 8000 — no conflict with ChromaDB (8001) or Streamlit (8501).")
    cors_origins: str = Field(
        default="*",
        description="Comma-separated allowed CORS origins. Use '*' for dev, restrict in prod (e.g. 'http://localhost:8501').",
    )

    # ── Streamlit UI ──────────────────────────────────────────────────────────
    streamlit_server_port: int = Field(
        default=8501,
        gt=0,
        description="Streamlit server port. Default 8501 — no conflict with API (8000) or ChromaDB (8001).",
    )
    streamlit_server_address: str = Field(
        default="localhost",
        description="Streamlit bind address.",
    )
    api_base_url: str = Field(
        default="http://localhost:8000",
        description="Full base URL of the FastAPI backend, used by the Streamlit UI to make HTTP calls.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Minimum log level. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )
    log_format: str = Field(
        default="console",
        description="Log renderer: 'json' for production, 'console' for development.",
    )


import os

# Module-level singleton — import this everywhere instead of re-instantiating.
settings = Settings()


def _set_env(var_name: str, value: str | None) -> None:
    """Set env var only when value is non-empty after stripping."""
    if value is None:
        return
    cleaned = value.strip()
    if cleaned:
        os.environ[var_name] = cleaned


# Push API keys to os.environ so that underlying SDKs (OpenAI, LlamaIndex, Cohere)
# can pick them up automatically since pydantic-settings does not export them.
_set_env("OPENAI_API_KEY", settings.openai_api_key)
_set_env("COHERE_API_KEY", settings.cohere_api_key)

# Normalize LangSmith/LangChain aliases to avoid accidental key mismatch.
# Prefer LANGCHAIN_* first because this project's .env convention uses it.
resolved_langsmith_key = (
    settings.langchain_api_key.strip() or settings.langsmith_api_key.strip()
)
_set_env("LANGSMITH_API_KEY", resolved_langsmith_key)
_set_env("LANGCHAIN_API_KEY", resolved_langsmith_key)

resolved_langsmith_endpoint = (
    settings.langchain_endpoint.strip() or settings.langsmith_endpoint.strip()
)
_set_env("LANGSMITH_ENDPOINT", resolved_langsmith_endpoint)
_set_env("LANGCHAIN_ENDPOINT", resolved_langsmith_endpoint)

_set_env("LANGCHAIN_PROJECT", settings.langchain_project)
_set_env("LANGSMITH_PROJECT", settings.langchain_project)

tracing_flag = "true" if settings.langchain_tracing_v2 else "false"
os.environ["LANGCHAIN_TRACING_V2"] = tracing_flag
os.environ["LANGSMITH_TRACING_V2"] = tracing_flag
