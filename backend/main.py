import os
import uuid
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag import RAGPipeline

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

rag = RAGPipeline()

# ── Serve frontend ──────────────────────────────────────────────────────────
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

@app.get("/")
def root():
    index = frontend_path / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "RAG Chatbot API — visit /docs"}


# ── Schemas ─────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    collection: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

class CollectionInfo(BaseModel):
    name: str
    doc_count: int
    files: List[str]


# ── Routes ───────────────────────────────────────────────────────────────────
@app.post("/upload", summary="Upload a PDF and index it into ChromaDB")
async def upload_pdf(
    file: UploadFile = File(...),
    collection: str = "default"
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_id = uuid.uuid4().hex[:8]
    safe_name = f"{file_id}_{file.filename}"
    dest = UPLOAD_DIR / safe_name

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = rag.ingest_pdf(str(dest), collection=collection, source_name=file.filename)
        return {
            "message": f"Indexed '{file.filename}' successfully.",
            "chunks": chunks,
            "collection": collection,
            "filename": file.filename,
        }
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, summary="Ask a question against indexed documents")
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = rag.query(req.question, collection=req.collection)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections", summary="List all collections and their document counts")
def list_collections():
    return rag.list_collections()


@app.delete("/collections/{name}", summary="Delete a collection")
def delete_collection(name: str):
    rag.delete_collection(name)
    return {"message": f"Collection '{name}' deleted."}


@app.get("/health")
def health():
    return {"status": "ok"}
