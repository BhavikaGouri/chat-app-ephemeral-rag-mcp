# Category-Aware Adaptive Ephemeral RAG-MCP Server

A dual-mode document QA system that combines a Streamlit chatbot interface with a FastMCP server, enabling intelligent retrieval-augmented generation with adaptive query strategies and zero-shot intent classification.

---

## Overview

This project implements a **Category-Aware Adaptive RAG** pipeline in two modes:

| Mode | Interface | Model |
|------|-----------|-------|
| Chatbot | Streamlit web app | Phi-3-mini via Ollama |
| MCP Server | FastMCP + Claude Desktop | Claude (via MCP tools) |

Both modes share a common `core/` module, ensuring consistent retrieval and classification logic across interfaces.

---

## Features

- **Zero-shot intent classification** using `facebook/bart-large-mnli` — automatically routes queries into `search`, `explanation`, or `generation` intents
- **Adaptive retrieval strategies** — top-3 for focused search, top-6 for broad explanation, MMR (Maximal Marginal Relevance) for generation tasks
- **Ephemeral ChromaDB** client — no persistent vector store; documents live only for the session
- **Multi-document session management** — upload and query across multiple PDFs in a single session
- **Five MCP tools** exposed to Claude Desktop for programmatic document interaction
- **`all-MiniLM-L6-v2` embeddings** via SentenceTransformers for fast, lightweight semantic search

---

## Architecture

```
rag-mcp-server/
├── core/                        # Shared logic (used by both interfaces)
│   ├── embedder.py              # SentenceTransformer: all-MiniLM-L6-v2
│   ├── retriever.py             # ChromaDB ephemeral client + adaptive retrieval
│   ├── classifier.py            # Zero-shot intent classification (BART-MNLI)
│   └── generator.py             # Response generation logic
├── streamlit_app/
│   └── app.py                   # Streamlit chatbot UI
├── mcp_server/
│   └── server.py                # FastMCP server with 5 MCP tools
├── claude_desktop_config.json   # Claude Desktop MCP configuration
├── pyproject.toml
└── README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector Store | ChromaDB (ephemeral client) |
| Intent Classification | `facebook/bart-large-mnli` (zero-shot) |
| Chatbot LLM | Phi-3-mini via Ollama |
| MCP Framework | FastMCP |
| Chatbot UI | Streamlit |
| MCP Client | Claude Desktop |
| Package Manager | `uv` |

---

## MCP Tools

The FastMCP server exposes five tools to Claude Desktop:

1. **`upload_document`** — Ingest a PDF or text file into the ephemeral vector store
2. **`query_documents`** — Run a RAG query with adaptive retrieval based on classified intent
3. **`list_documents`** — List all documents currently loaded in the session
4. **`get_chunk`** — Retrieve a specific chunk by ID for inspection
5. **`clear_session`** — Reset the ephemeral store and clear all loaded documents

---

## Retrieval Strategy

Intent is classified before retrieval, and the strategy adapts accordingly:

| Classified Intent | Retrieval Strategy | Rationale |
|-------------------|--------------------|-----------|
| `search` | Top-3 chunks | Precise, focused lookup |
| `explanation` | Top-6 chunks | Broader context needed |
| `generation` | MMR (top-6) | Diverse, non-redundant coverage |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for environment management
- [Ollama](https://ollama.com/) installed and running (for Streamlit mode)
- Claude Desktop (for MCP mode)

### 1. Clone the Repository

```bash
git clone https://github.com/BhavikaGouri/rag-mcp-server.git
cd rag-mcp-server
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Pull the LLM (Streamlit mode only)

```bash
ollama pull phi3:mini
```

---

## Running the Project

### Streamlit Chatbot

```bash
uv run streamlit run streamlit_app/app.py
```

Open `http://localhost:8501`, upload your documents, and start querying.

### MCP Server (Claude Desktop)

1. Update `claude_desktop_config.json` with the absolute path to your virtual environment Python and `server.py`:

```json
{
  "mcpServers": {
    "rag-mcp": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/mcp_server/server.py"],
      "cwd": "/absolute/path/to/rag-mcp-server"
    }
  }
}
```

2. Copy the config to the Claude Desktop config directory and restart Claude Desktop.

3. The five MCP tools will now be available in your Claude Desktop sessions.

---

## How It Works

```
User Query
    │
    ▼
Intent Classifier (BART-MNLI)
    │
    ├── search      → Top-3 retrieval
    ├── explanation → Top-6 retrieval
    └── generation  → MMR retrieval
              │
              ▼
    ChromaDB Ephemeral Store
    (all-MiniLM-L6-v2 embeddings)
              │
              ▼
    Retrieved Chunks + Query
              │
    ┌─────────┴──────────┐
    │                    │
Phi-3-mini           Claude Desktop
(Streamlit)          (MCP Tools)
    │                    │
    ▼                    ▼
  Response            Response
```

---

## Key Design Decisions

- **Ephemeral over persistent** — ChromaDB's in-memory client was chosen to keep sessions stateless and avoid stale index issues across runs
- **Shared `core/`** — Prevents logic drift between the two interfaces; both always use the same embedder, retriever, and classifier
- **Absolute paths in MCP config** — Required by Claude Desktop's process spawning; relative paths cause silent failures
- **`uv` for environment management** — Fast, reproducible installs with a single lockfile

---

## Troubleshooting

**MCP server not showing up in Claude Desktop**
- Ensure all paths in `claude_desktop_config.json` are absolute
- Confirm the `cwd` field points to the project root
- Restart Claude Desktop after any config change

**Ollama model not found**
- Run `ollama list` to verify `phi3:mini` is pulled
- Ensure the Ollama daemon is running (`ollama serve`)

**ChromaDB import error**
- Run `uv sync` to ensure all dependencies are installed in the venv

---

## Project Status

This project was built as a portfolio piece demonstrating:
- MCP protocol integration with Claude Desktop
- Adaptive RAG with intent-aware retrieval
- Dual-interface architecture with shared core logic

---

## Author

**Bhavika Gouri**  
Production Engineering Student, Sharda University  
GitHub: [@BhavikaGouri](https://github.com/BhavikaGouri
