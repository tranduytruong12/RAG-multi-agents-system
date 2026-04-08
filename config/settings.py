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

    # ── FastAPI ───────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="API server bind host.")
    api_port: int = Field(default=8000, gt=0, description="API server bind port.")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Minimum log level. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )
    log_format: str = Field(
        default="console",
        description="Log renderer: 'json' for production, 'console' for development.",
    )


# Module-level singleton — import this everywhere instead of re-instantiating.
settings = Settings()
