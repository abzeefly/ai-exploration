# AI Exploration

Demos exploring modern AI tooling for RAG applications.

## Demos

### Demo 1 — Arize Phoenix: RAG Observability (`demo1-phoenix-rag/`)

A RAG pipeline over 2024 tax policy documents, fully instrumented with [Arize Phoenix](https://phoenix.arize.com) for end-to-end tracing and evaluation.

**Stack:** Ollama (llama3.2 + nomic-embed-text) · ChromaDB · Arize Phoenix · LangChain

**RAG features demonstrated:**
| Feature | How |
|---|---|
| End-to-end tracing | Every span traced: embed → retrieve → generate |
| Custom span attributes | Per-chunk metadata, token counts, source files on each span |
| Hallucination evaluation | LLM-as-judge checks if answer is grounded in retrieved context |
| QA correctness | Judge compares answer against a reference answer |
| Retrieval relevance | Judge checks if retrieved chunks are relevant to the query |
| Evaluation upload | Eval results posted back to Phoenix as span annotations |

**Quick start:**
```bash
cd demo1-phoenix-rag
cp .env.example .env
make install                # sets up .venv with Poetry 2.3+
docker compose up -d
# Wait ~2 min for Ollama to pull nomic-embed-text
docker compose logs -f rag-app
```

Phoenix UI: http://localhost:6006

**Notebooks:**
```bash
source .venv/bin/activate
jupyter notebook notebooks/
# 01_ingest_and_embed.ipynb  — load, chunk, embed, store
# 02_rag_evaluations.ipynb   — hallucination, QA, retrieval relevance evals
```

---

### Demo 2 — MLflow AI Gateway: Model Router + RAG Evaluation (`demo2-mlflow-gateway/`)

MLflow AI Gateway routing between two Ollama models (llama3.2 3B vs mistral 7B), plus `mlflow.evaluate()` for RAG quality measurement.

**Stack:** MLflow AI Gateway (2.22.x) · Ollama (llama3.2 + mistral + nomic-embed-text) · MLflow Tracking

**RAG features demonstrated:**
| Feature | How |
|---|---|
| Provider abstraction | Same call, different provider by changing route name |
| Embeddings via gateway | nomic-embed-text routed through unified API |
| RAG faithfulness eval | `mlflow.evaluate()` with `faithfulness` metric |
| Answer relevance eval | `mlflow.evaluate()` with `answer_relevance` metric |
| Context relevance eval | `mlflow.evaluate()` with `relevance` metric |
| llama3.2 vs mistral comparison | Two MLflow runs, same judge, compare quality + latency |
| MLflow experiment tracking | All runs logged, compare in MLflow UI |

**Quick start:**
```bash
cd demo2-mlflow-gateway
cp .env.example .env
make install                # sets up .venv with Poetry 2.3+
docker compose up -d
# Wait ~5 min for Ollama to pull llama3.2 + mistral + nomic-embed-text (~6 GB total)
docker compose run --rm demo-app            # routing demo
docker compose run --rm demo-app python -m app.rag_eval   # RAG eval
```

Gateway routes: http://localhost:5000/api/2.0/gateway/routes/  
MLflow UI (after running evals): `mlflow ui --port 5001`

**Notebooks:**
```bash
source .venv/bin/activate
jupyter notebook notebooks/
# 01_gateway_demo.ipynb   — routing, embeddings, latency benchmarking
# 02_rag_evaluation.ipynb — faithfulness/relevance evals, llama3.2 vs mistral
```

---

## Local Dev Setup

Both demos use **Poetry 2.3+** with in-project `.venv`. Because VS Code sets `VIRTUAL_ENV`
to the system Python, use `make install` (not bare `poetry install`) — the Makefile clears
the stale variable before creating the venv.

```bash
cd demo1-phoenix-rag   # or demo2-mlflow-gateway
make install           # creates .venv and installs all deps
source .venv/bin/activate
```

## Prerequisites

- Docker Desktop with Compose v2 (`docker compose version`)
- ~6 GB disk space for Ollama models (llama3.2 ~2 GB, mistral ~4 GB, nomic-embed-text ~274 MB)
- Python 3.13 (Homebrew) — already present on your machine
