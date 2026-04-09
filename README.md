# Multi-Agent RAG Customer Support System

A **production-ready, multi-agent Retrieval-Augmented Generation system** for customer support automation, built with **LangGraph**, **LlamaIndex**, and **ChromaDB**.

## Architecture Overview

```
User Query (POST /chat)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                     OrchestratorAgent (LangGraph DAG)         │
│                                                              │
│  ┌─────────────┐    ┌────────────┐    ┌────────────────────┐ │
│  │  Intent     │───▶│  Retriever │───▶│   DraftWriter      │ │
│  │  Classifier │    │  Agent     │    │   Agent (GPT-4o)   │ │
│  └─────────────┘    └────────────┘    └────────────────────┘ │
│                                                  │           │
│                                                  ▼           │
│                                        ┌──────────────────┐  │
│                                        │   QA Agent       │  │
│                                        │  (verify reply)  │  │
│                                        └──────────────────┘  │
│                                                  │           │
│                              ┌───────────────────┴─────────┐ │
│                              │  passed?                     │ │
│                         YES──┤                     NO (retry│ │
│                              │         or Human Review)     │ │
│                              └──────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
Final Reply (JSON response)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | **LangGraph** (Cyclic graph + HITL checkpointing) |
| RAG Framework | **LlamaIndex** |
| LLM | **OpenAI GPT-4o** (Ollama is optional) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | **ChromaDB** (self-hosted) |
| API | **FastAPI** + Uvicorn |
| Config | Pydantic Settings v2 |
| Logging | **structlog** (JSON in prod, colored in dev) |
| Package Manager | **Poetry** |
| Container | Docker + Docker Compose |

---

## Project Structure

```
customer_support_rag/
├── pyproject.toml          # Poetry project + all dependencies
├── .env.example            # Environment variable template
├── Makefile                # Developer workflow commands
│
├── config/
│   └── settings.py         # Pydantic BaseSettings (loads from .env)
│
├── core/
│   ├── types.py            # SHARED CONTRACTS: SupportState, QAVerdict, etc.
│   ├── exceptions.py       # Custom exception hierarchy
│   └── logging.py          # structlog configuration
│
├── ingestion/
│   ├── loaders.py          # Markdown & PDF document loaders
│   ├── chunkers.py         # Semantic chunking (SentenceSplitter)
│   └── pipeline.py         # End-to-end ingestion orchestrator
│
├── retrieval/              # Phase 1B
│   ├── vector_store.py     # ChromaDB manager
│   ├── retriever.py        # Hybrid dense+sparse retrieval
│   └── ranker.py           # Re-ranking / relevance scoring
│
├── agents/                 # Phase 2
│   ├── orchestrator.py     # LangGraph StateGraph DAG
│   ├── intent_classifier.py
│   ├── drafter.py
│   └── qa_agent.py
│
├── api/                    # Phase 2
│   ├── main.py             # FastAPI app factory
│   └── routes/
│       ├── chat.py         # POST /chat
│       └── ingest.py       # POST /ingest
│
├── tests/
│   ├── test_types.py
│   ├── test_loaders.py
│   ├── test_chunkers.py
│   ├── test_retriever.py
│   ├── test_agents.py
│   └── test_api.py
│
├── data/
│   └── docs/               # Place source documents here (.md, .pdf)
│
├── Dockerfile
└── docker-compose.yml
```

---

## Quickstart

### 1. Install Dependencies
```bash
make install
# Then edit .env with your OPENAI_API_KEY
```

### 2. Add Source Documents
```bash
mkdir -p data/docs
# Copy your .md and .pdf support documents into data/docs/
```

### 3. Ingest Documents
```bash
make ingest
```

### 4. Start the API
```bash
make dev
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 5. Query the System
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "I need a refund for my last order", "session_id": "user-123"}'
```

### 6. UI
```bash

---

## Docker (Full Stack)

```bash
make docker-up    # Starts API + ChromaDB
make docker-down  # Stops all services
```

---

## Human-in-the-Loop (HITL)

The LangGraph orchestrator supports **interrupt-and-resume** for cases where the QA agent flags a reply as requiring human review:

1. QA Agent sets `state["requires_human_review"] = True`
2. LangGraph checkpoints the state and raises `HumanReviewRequired`
3. A human reviews and sets `state["human_feedback"]`
4. The graph resumes from the checkpoint with the human's feedback

---

## Testing

```bash
make test          # Full suite with coverage
make test-fast     # Fast run (fail-on-first, no coverage)
```

---

## Environment Variables

See [`.env.example`](.env.example) for all configurable options.

---

## 🗂️ Project Roadmap — Sprints & Milestones

### 📊 High-Level Timeline

| Sprint | Phase | Focus | Status |
|--------|-------|-------|--------|
| Sprint 0 | SETUP | Environment bootstrap | ✅ Done |
| Sprint 1 | RAG | Ingestion pipeline implementation | ✅ Done |
| Sprint 2 | RAG | Retrieval & vector search | ✅ Done |
| Sprint 3 | MULTI-AGENT | Multi-agent DAG (LangGraph) | ✅ Done |
| Sprint 4 | BACK-END | API, HITL & test suite | ✅ Done |
| Sprint 5 | HARDENING | Docker, retry logic & polish | 🔲 Pending |

---

### ✅ Sprint 0 — Environment Bootstrap *(Week 0 — COMPLETE)*

**Goal:** Working project skeleton with all contracts importable.

#### Tasks
- [x] Generate project scaffold (14 files from Prompt 1)
- [x] Define `SupportState`, `QAVerdict`, `DocumentChunk`, `RetrievalResult` in `core/types.py`
- [x] Define full exception hierarchy in `core/exceptions.py`
- [x] Configure `structlog` in `core/logging.py`
- [x] Write `config/settings.py` with all environment variables
- [x] Write `pyproject.toml` with all Phase 1–3 dependencies
- [x] Create `Makefile`, `.env.example`, `README.md`

#### 🏁 Milestone — Sprint 0 Done When:
```bash
poetry install                    # exits 0
cp .env.example .env              # fill OPENAI_API_KEY
python -c "
from core.types import SupportState, QAVerdict
from core.exceptions import RAGBaseException
from core.logging import get_logger
from config.settings import settings
print('✅ All contracts importable')
print(f'   LLM: {settings.llm_model_name}')
print(f'   Embed: {settings.embed_model_name}')
"
```
---

### ✅ Sprint 1 — Ingestion Pipeline *(Week 1 — Prompt 1 TODOs)*

**Goal:** Load real `.md` and `.pdf` documents from disk, chunk them into `DocumentChunk` objects. No vector storage yet.

#### Framework Boundary
> 🔵 **LlamaIndex only** — `SimpleDirectoryReader`, `PDFReader`, `SentenceSplitter`

#### Tasks

**`ingestion/loaders.py`**
- [x] Implement `MarkdownLoader.load_directory(directory)` using `rglob("*.md")` + skip-on-error
- [x] Implement `PDFLoader.load(path)` using `PDFReader().load_data(file=path)` (page-per-doc)
- [x] Implement `TextLoader.load_directory(directory)` using `rglob("*.txt")` + skip-on-error
- [x] Wrap all loader calls in `asyncio.to_thread()` to avoid blocking event loop
- [x] Enrich every `doc.metadata` with `source_path`, `file_type`, `loaded_at`
- [x] Wrap exceptions → `LoaderError` with structured context dict

**`ingestion/chunkers.py`**
- [x] Implement `SemanticChunker.chunk(documents)` using `SentenceSplitter`
- [x] Run `splitter.get_nodes_from_documents()` via `asyncio.to_thread()`
- [x] Convert each `TextNode` → `DocumentChunk(chunk_id, source_path, content, metadata, token_count)`
- [x] Implement `_estimate_tokens()` with tiktoken (`cl100k_base` encoding) for accuracy
- [x] Implement `SemanticChunker.chunk_single(document)` as delegate to `chunk()`
- [x] Wrap exceptions → `ChunkerError`

**`ingestion/pipeline.py`** *(load + chunk phases only)*
- [x] Implement `IngestionPipeline._load_all_documents()` — discover + load `.md`, `.txt` and `.pdf` files
- [x] Implement `IngestionPipeline.run()` load & chunk phases
- [x] Collect `failed_documents` list for files that error during loading
- [x] Return `IngestionResult` with counts and duration (leave embed+store as TODO)
- [x] Add structured logging throughout (start, chunk_complete, failures)

**Tests**
- [x] `tests/test_loaders.py` — test MarkdownLoader, TextLoader and PDFLoader with fixture files
- [x] `tests/test_chunkers.py` — assert chunk count, content type, metadata presence
- [x] Add some samples `.md`, `.txt` and `.pdf` files to `data/docs/` for manual testing

#### 🏁 Milestone — Sprint 1 Done When:
```bash
# Place a test doc in data/docs/test.md, then:
pytest tests/test_loaders.py tests/test_chunkers.py -v   # all pass
```

---

### ✅ Sprint 2 — Retrieval & Vector Search *(Week 2 — Prompt 2)*

**Goal:** Connect ChromaDB, embed chunks, and retrieve relevant context for any query using hybrid search.

#### Framework Boundary
> 🔵 **LlamaIndex only** — `VectorStoreIndex`, `ChromaVectorStore`, `BM25Retriever`, `QueryFusionRetriever`

#### Tasks

**`retrieval/vector_store.py`**
- [x] Implement `VectorStoreManager.connect()` — init `chromadb.HttpClient`, wrap in `ChromaVectorStore` + `StorageContext`
- [x] Implement `VectorStoreManager.add_nodes(nodes)` — build `VectorStoreIndex` from `TextNode` list with OpenAI embeddings
- [x] Implement `VectorStoreManager.delete_collection()` — drop and recreate collection for full re-ingest
- [x] Wire into `IngestionPipeline.run()` embed+store step

**`retrieval/retriever.py`**
- [x] Implement `HybridRetriever.build_index()` — load `VectorStoreIndex` from existing ChromaDB collection
- [x] Implement `HybridRetriever.retrieve(query, top_k)`:
  - Dense path: `VectorIndexRetriever` (cosine similarity)
  - Sparse path: `BM25Retriever` on stored nodes
  - Fusion: `QueryFusionRetriever` with `mode="reciprocal_rerank"` (RRF)
- [x] Convert `NodeWithScore` list → `list[RetrievalResult]` (no LlamaIndex types leak out)

**`retrieval/reranker.py`**
- [x] Implement `ContextRanker.rank()` — LLM-based relevance re-scoring or cross-encoder

**Tests**
- [x] `tests/test_retriever.py` — mock `VectorStoreManager`, assert `HybridRetriever` returns `list[RetrievalResult]`

#### 🏁 Milestone — Sprint 2 Done When:
```bash
pytest tests/test_retriever.py -v   # all pass
```

---

### ✅ Sprint 3 — Multi-Agent DAG *(Week 3 — Prompt 3A)*

**Goal:** Build the full LangGraph state machine. All 4 agents implemented and connected. QA retry loop working.

#### Framework Boundary
> 🟠 **LangGraph + LangChain** — `StateGraph`, `ChatOpenAI`, `with_structured_output()`
> 🔵 **LlamaIndex** called only inside the `retrieve` node

#### Tasks

**`agents/intent_classifier.py`**
- [x] Implement `IntentClassifierAgent.classify(query)` using `ChatOpenAI` + few-shot `ChatPromptTemplate`
- [x] Parse LLM output → validate against `VALID_INTENTS`, raise `IntentClassificationError` if invalid
- [x] Add tenacity retry decorator (`@retry(stop=stop_after_attempt(3)`)

**`agents/drafter.py`**
- [x] Implement `DraftWriterAgent.draft(state)`:
  - [x] Build context string from `state["retrieved_context"]`
  - [x] Use prompt: `"Given context: {context}\nIntent: {intent}\nDraft a professional reply to: {query}"`
  - [x] Call `ChatOpenAI(model=settings.llm_model_name, temperature=0.0)`
  - [x] If `state["human_feedback"]` is set, incorporate it into prompt

**`agents/qa_agent.py`**
- [x] Implement `QAAgent.verify(state)` using `ChatOpenAI.with_structured_output(QAVerdict)`
- [x] Prompt checks: tone, accuracy vs retrieved context, policy completeness
- [x] Return fully populated `QAVerdict` Pydantic model

**`agents/orchestrator.py`**
- [x] Define LangGraph `StateGraph` with `SupportState` schema
- [x] Add nodes: `classify_intent`, `retrieve`, `draft_reply`, `qa_check`, `finalize`, `escalate`
- [x] Add edges:
  - `classify_intent → retrieve → draft_reply → qa_check`
  - Conditional: `qa_check → draft_reply` (if not passed AND retry_count < max)
  - Conditional: `qa_check → escalate` (if retry_count >= max)
  - Conditional: `qa_check → finalize` (if passed)
- [x] Compile graph: `graph = builder.compile(checkpointer=MemorySaver())`
- [x] Implement `OrchestratorAgent.run(state)` — invoke compiled graph

**Tests**
- [x] `tests/test_agents.py` — test state transitions with all sub-agents mocked

#### 🏁 Milestone — Sprint 3 Done When:
```bash
pytest tests/test_agents.py -v   # all pass
```

---

### ✅ Sprint 4 — API, HITL & Test Suite *(Week 4 — Prompt 3B)*

**Goal:** HTTP API live and fully tested. Human-in-the-loop interrupt/resume working.

#### Tasks

**`api/main.py`**
- [x] FastAPI app with `lifespan` context: call `configure_logging()` on startup
- [x] CORS middleware, global exception handler for `APIError` → HTTP response
- [x] Mount `/chat` and `/ingest` routers

**`api/routes/chat.py`**
- [x] `POST /chat` — validate input, bind `request_id` to log context, invoke `OrchestratorAgent.run()`
- [x] `GET /health` — return `{"status": "ok", "version": "0.1.0"}`
- [x] Handle `GraphInterrupt` → return `202 Accepted` with state snapshot

**`api/routes/ingest.py`**
- [x] `POST /ingest` — accept optional `source_dir`, run `IngestionPipeline.run()` as FastAPI `BackgroundTask`
- [x] `GET /ingest/status/{job_id}` — return job status (simple in-memory dict)

**Human-in-the-Loop (HITL)**
- [x] `interrupt()` call in `qa_check` node when `requires_human_review=True` (Sprint 3)
- [x] `POST /chat/resume` endpoint — accept `session_id` + `human_feedback`, call `graph.ainvoke(Command(resume=feedback))`

**Test Suite**
- [x] `tests/test_types.py` — roundtrip `SupportState` and `QAVerdict` serialization
- [x] `tests/test_api.py` — FastAPI `TestClient` for all endpoints, mock orchestrator
- [x] Achieved **92% test coverage** (target: >80%) — 114/114 tests passing

#### 🏁 Milestone — Sprint 4 Done When:
```bash
make dev &   # API starts on :8000
curl -s -X POST http://localhost:8000/health | python -m json.tool
# → {"status": "ok", ...}

curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"I need a refund","session_id":"s-001"}' | python -m json.tool
# → {"reply": "...", "intent": "refund", "metadata": {...}}

make test
# → All tests pass, coverage > 80%
```

---

### 🔲 Sprint 5 — Docker, Hardening & Production Polish *(Week 5)*

**Goal:** Fully containerized. Retry logic. Structured logging in JSON. Ready to demo.

#### Tasks

**Containerization**
- [ ] Write multi-stage `Dockerfile` (`builder` → `runtime` slim image)
- [ ] Write `docker-compose.yml` with `app`, `chroma` services
- [ ] Add health checks to compose services
- [ ] Test `make docker-up` end-to-end

**Hardening**
- [ ] Add `tenacity` retry wrappers to all LLM calls (`stop_after_attempt(3)`, `wait_exponential`)
- [ ] Add `tenacity` retry to ChromaDB connect in `VectorStoreManager`
- [ ] Set `LOG_FORMAT=json` in Docker compose — verify structured JSON output
- [ ] Add `request_id` to all API responses in response headers

**Final Validation**
- [ ] All 6 test files pass with >80% coverage
- [ ] `docker compose up --build` works from a clean checkout
- [ ] End-to-end test: POST /ingest → POST /chat → meaningful reply

#### 🏁 Milestone — Sprint 5 (PROJECT COMPLETE) When:
```bash
make docker-up

# Full end-to-end:
curl -X POST http://localhost:8000/ingest   # returns job accepted
sleep 30                                    # wait for ingestion

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What is your refund policy?","session_id":"demo-1"}'
# → structured JSON reply grounded in your docs

make docker-down
```

---

### 📈 Success Metrics

| Metric | Target |
|--------|--------|
| Test coverage | > 80% on non-stub code |
| Intent classification accuracy | > 90% on test queries |
| QA pass rate (first attempt) | > 75% |
| API response time (p95) | < 5 seconds |
| Docker image size | < 600 MB |
| End-to-end ingest time (100 docs) | < 2 minutes |

---

### 🔗 Sprint Dependencies

```
Sprint 0 (scaffold)
    └── Sprint 1 (loaders + chunkers)
            └── Sprint 2 (embed + vector store + retriever)
                    └── Sprint 3 (agents + LangGraph DAG)
                            └── Sprint 4 (API + HITL + tests)
                                    └── Sprint 5 (Docker + hardening)
```

> Each sprint's milestone command must pass before starting the next sprint.
