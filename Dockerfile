# =============================================================================
# FILE: customer_support_rag/Dockerfile
# Multi-stage build — keeps the final image lean and secure.
#
# Stage 1 (builder): Install Poetry + export a plain requirements.txt, then
#   install all production dependencies into /opt/venv so we can copy only
#   the binaries to the runtime image (avoids shipping Poetry itself).
#
# Stage 2 (runtime): python:3.12-slim — copies the pre-built venv and source,
#   runs as a non-root user.
#
# Build:  docker build -t customer-support-rag .
# Run:    docker compose up
# =============================================================================

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System deps needed to compile some Python packages (e.g. chromadb, grpcio)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (pinned version for reproducibility)
ENV POETRY_VERSION=1.8.3
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /build

# Copy dependency manifests first (better layer caching)
COPY pyproject.toml poetry.lock* ./

# Export production-only requirements to a plain requirements.txt so the
# runtime stage can install with pip (no Poetry needed at runtime).
RUN poetry export \
        --without dev \
        --without-hashes \
        --format requirements.txt \
        -o requirements.txt

# Install into an isolated virtualenv inside /opt/venv
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt \
    && /opt/venv/bin/pip install --no-cache-dir llama-index-llms-openai


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Sane Python defaults for containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Minimal runtime OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy pre-built virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source (respects .dockerignore)
COPY --chown=appuser:appgroup . .

# Create data directories (persist via volume mounts in compose)
RUN mkdir -p /app/data/docs /app/data/chroma \
    && chown -R appuser:appgroup /app/data

USER appuser

# ── Health check (used by Docker Compose depends_on) ──────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Default entrypoint: FastAPI backend ───────────────────────────────────────
# Override with 'streamlit run ui/app.py ...' for the UI service.
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
