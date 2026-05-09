# 🧠 RAG Chatbot

A production-ready Retrieval-Augmented Generation chatbot — upload PDFs, ask questions, get answers with source citations.

**Stack:** LangChain · ChromaDB · FastAPI · Vanilla JS

---

## Features

- 📄 Upload multiple PDFs (drag & drop or click)
- 🔍 Semantic search with vector embeddings
- 💬 LLM-generated answers grounded in your documents
- 📚 Source citations with page numbers and snippets
- 📁 Multiple named collections (separate knowledge bases)
- 🌙 Dark mode UI, fully responsive
- 🐳 Docker-ready for deployment

---

## Quick start

### 1. Clone & set up environment

```bash
cd rag-chatbot
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY (or leave blank for free local mode)
```

### 2. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Run the server

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## LLM Options

### Option A — OpenAI (recommended)

Add your key to `.env`:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini   # cheap and fast
```

### Option B — Free local with Ollama (no API key)

1. Install [Ollama](https://ollama.com)
2. Pull a model: `ollama pull mistral`
3. Leave `OPENAI_API_KEY` blank in `.env`

The app auto-detects which mode to use.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload & index a PDF |
| `POST` | `/chat` | Ask a question |
| `GET` | `/collections` | List all collections |
| `DELETE` | `/collections/{name}` | Delete a collection |
| `GET` | `/health` | Health check |

Interactive docs: **http://localhost:8000/docs**

---

## Docker deployment

```bash
docker build -t rag-chatbot .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/chroma_db:/app/backend/chroma_db \
  rag-chatbot
```

---

## Deploy to Railway / Render / Fly.io

1. Push to GitHub
2. Connect repo in Railway/Render
3. Set `OPENAI_API_KEY` as an environment variable
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add a persistent disk mounted at `/app/backend/chroma_db`

---

## Project structure

```
rag-chatbot/
├── backend/
│   ├── main.py          # FastAPI routes
│   ├── rag.py           # LangChain + ChromaDB pipeline
│   └── requirements.txt
├── frontend/
│   └── index.html       # Single-page chat UI
├── Dockerfile
├── .env.example
└── README.md
```

---

## How it works

```
PDF Upload → PyMuPDF parse → RecursiveTextSplitter (800 chars)
    → Embeddings (OpenAI / MiniLM) → ChromaDB store

User Question → Embed question → ChromaDB similarity search (top 4)
    → LangChain RetrievalQA → LLM answer + source citations
```

---

## Resume talking points

- Built a **production RAG pipeline** with LangChain + ChromaDB + FastAPI
- Implemented **semantic search** using OpenAI / HuggingFace embeddings
- Designed a **multi-collection** architecture for isolated knowledge bases
- Added **source citation** with page-level provenance
- Containerized with Docker and deployed to cloud platforms
