# RAG Research Agent 🧠🔎

A hybrid **Retrieval-Augmented Generation (RAG)** + **tool-use** agent you can run locally.  
It can:
- Retrieve from your local documents (FAISS + sentence-transformers).
- Use tools like **Wikipedia search** and **on-the-fly plotting** from CSV.
- Serve a simple **FastAPI** endpoint for chatting.
- Include **tests**, **Dockerfile**, and **CI**.

## Quickstart

```bash
# Clone & install
uv venv .venv && source .venv/bin/activate  # or python -m venv .venv
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and set OPENAI_API_KEY if you want LLM responses via OpenAI.
# If not set, the agent will fall back to a minimal offline template answer.

# Ingest sample docs & build index
python scripts/ingest.py --data-dir data/sample_docs --index-dir data/index

# Run API
uvicorn src.app.main:app --reload
```

Open: http://localhost:8000/docs

## Project layout

```
rag-research-agent/
├── src/
│   ├── agent/
│   │   ├── rag.py
│   │   ├── tools.py
│   │   └── agent.py
│   └── app/
│       └── main.py
├── scripts/
│   └── ingest.py
├── data/
│   └── sample_docs/    # example content
├── tests/
│   └── test_retrieval.py
├── docs/
│   └── LEARNING_LOG.md
├── .github/workflows/ci.yml
├── requirements.txt
├── Dockerfile
├── Makefile
├── .env.example
└── README.md
```

## Architecture

```mermaid
flowchart LR
    U[User] -->|query| API[FastAPI /chat]
    API --> AG[Agent]
    AG --> RAG[Retriever (FAISS + ST)]
    RAG --> IDX[(Vector Index)]
    AG --> TOOLS[Tools]
    TOOLS --> WIKI[Wikipedia]
    TOOLS --> PLOT[Plot CSV]
    AG --> LLM[LLM (OpenAI or Offline)]
    LLM --> AG
    AG -->|answer + citations| API
```

## Notable concepts
- **Chunking & embeddings:** `sentence-transformers/all-MiniLM-L6-v2` into **FAISS**.
- **Hybrid retrieval hooks:** place for BM25 or metadata filters.
- **Tool use:** simple registry; tools are selected via heuristic or model hints.
- **Reproducibility:** `scripts/ingest.py` and unit tests.
- **Bonus:** includes README, diagram, Docker, CI, and a learning log template.

## Evaluation (optional)
See `notebooks/` placeholder and `tests/` for a starting point. You can add datasets like Q/A pairs and compute retrieval precision@k.

## License
MIT
