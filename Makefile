# FILE: customer_support_rag/Makefile
.PHONY: install dev ui test lint format type-check ingest docker-build docker-up docker-down docker-logs clean help

# ── Default ──────────────────────────────────────────────────────────────────
help:
	@echo "Multi-Agent RAG Customer Support System"
	@echo "======================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install       Install all dependencies via Poetry"
	@echo "  dev           Start FastAPI dev server with hot reload"
	@echo "  ui            Start Streamlit UI (port 8501)"
	@echo "  test          Run full test suite with coverage"
	@echo "  lint          Run ruff linter"
	@echo "  format        Auto-format code with ruff"
	@echo "  type-check    Run mypy type checker"
	@echo "  ingest        Run ingestion pipeline on ./data/docs/"
	@echo "  docker-build  Build Docker images without starting"
	@echo "  docker-up     Start all services via Docker Compose"
	@echo "  docker-down   Stop all Docker services"
	@echo "  docker-logs   Tail all Docker service logs"
	@echo "  clean         Remove __pycache__, .pytest_cache, htmlcov"

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	poetry install
	cp -n .env.example .env || true
	@echo "✅ Dependencies installed. Edit .env with your API keys."

# ── Development ───────────────────────────────────────────────────────────────
dev:
	poetry run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	poetry run streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0

db:
	poetry run chromadb run --host localhost --port 8001
# ── Testing ───────────────────────────────────────────────────────────────────
test:
	poetry run pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

test-fast:
	poetry run pytest tests/ -v -x --no-cov

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	poetry run ruff check .

format:
	poetry run ruff format .
	poetry run ruff check --fix .

type-check:
	poetry run mypy . --ignore-missing-imports

# ── Ingestion ─────────────────────────────────────────────────────────────────
ingest:
	poetry run python -c "
import asyncio
from pathlib import Path
from ingestion.pipeline import IngestionPipeline
async def main():
    pipeline = IngestionPipeline()
    result = await pipeline.run(Path('./data/docs'))
    print(f'Ingestion complete: {result}')
asyncio.run(main())
"

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-up:
	docker compose up --build -d
	@echo "✅ Services started."
	@echo "   API:      http://localhost:8000"
	@echo "   API docs: http://localhost:8000/docs"
	@echo "   ChromaDB: http://localhost:8001"
	@echo "   UI:       http://localhost:8501"

docker-down:
	docker compose down -v

docker-logs:
	docker compose logs -f

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleaned up build artifacts."
