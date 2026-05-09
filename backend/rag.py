"""
RAG pipeline — LangChain + ChromaDB + OpenAI (or HuggingFace fallback)
"""

import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Embeddings: prefer OpenAI, fall back to free HuggingFace ────────────────
def _get_embeddings():
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and openai_key != "sk-...":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(openai_api_key=openai_key)
    # Free local fallback — downloads ~90 MB once
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── LLM: prefer OpenAI, fall back to local Ollama ───────────────────────────
def _get_llm():
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and openai_key != "sk-...":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            openai_api_key=openai_key,
        )
    # Free local fallback — requires `ollama run mistral` running locally
    from langchain_community.llms import Ollama
    return Ollama(model=os.getenv("OLLAMA_MODEL", "mistral"))


RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that answers questions based strictly on the provided context.
If the answer is not in the context, say "I don't have enough information in the uploaded documents to answer this."

Context:
{context}

Question: {question}

Answer (be concise and cite which document/page you're drawing from when possible):""",
)

CHROMA_DIR = Path("chroma_db")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


class RAGPipeline:
    def __init__(self):
        self.embeddings = _get_embeddings()
        self.llm = _get_llm()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # Persistent Chroma client
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        # Cache: collection_name → Chroma vectorstore
        self._stores: Dict[str, Chroma] = {}

    # ── Internal helpers ────────────────────────────────────────────────────

    def _get_store(self, collection: str) -> Chroma:
        if collection not in self._stores:
            self._stores[collection] = Chroma(
                client=self._client,
                collection_name=collection,
                embedding_function=self.embeddings,
            )
        return self._stores[collection]

    # ── Public API ──────────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: str, collection: str = "default", source_name: str = "") -> int:
        """Load a PDF, split into chunks, embed, and store in ChromaDB."""
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()

        # Tag each page with the human-readable filename
        for doc in pages:
            doc.metadata["source"] = source_name or Path(pdf_path).name

        chunks = self.splitter.split_documents(pages)
        store = self._get_store(collection)
        store.add_documents(chunks)
        return len(chunks)

    def query(self, question: str, collection: str = "default", k: int = 4) -> Dict[str, Any]:
        """Retrieve relevant chunks and generate an answer with citations."""
        store = self._get_store(collection)
        retriever = store.as_retriever(search_kwargs={"k": k})

        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": RAG_PROMPT},
        )

        result = chain.invoke({"query": question})
        answer = result["result"]
        source_docs = result.get("source_documents", [])

        # Deduplicate sources for the citation panel
        seen = set()
        sources = []
        for doc in source_docs:
            meta = doc.metadata
            key = (meta.get("source", ""), meta.get("page", ""))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": meta.get("source", "Unknown"),
                    "page": meta.get("page", "?"),
                    "snippet": doc.page_content[:220].replace("\n", " ") + "…",
                })

        return {"answer": answer, "sources": sources}

    def list_collections(self) -> List[Dict]:
        cols = self._client.list_collections()
        result = []
        for col in cols:
            c = self._client.get_collection(col.name)
            result.append({
                "name": col.name,
                "doc_count": c.count(),
                "files": list({
                    m.get("source", "?")
                    for m in (c.get(include=["metadatas"])["metadatas"] or [])
                }),
            })
        return result

    def delete_collection(self, name: str):
        self._client.delete_collection(name)
        self._stores.pop(name, None)
