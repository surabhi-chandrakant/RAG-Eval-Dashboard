"""
engine.py — LlamaIndex RAG engine using Groq via OpenAI-compatible endpoint.

PDF/DOCX text extraction is done directly with pypdf and docx2txt,
bypassing SimpleDirectoryReader which falls back to raw bytes when
llama-index-readers-file is not installed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

GROQ_API_BASE = "https://api.groq.com/openai/v1"


# ── File → Document loader ────────────────────────────────

def _load_file(path: Path) -> list[Document]:
    """
    Extract clean text from PDF, DOCX, TXT, or MD.
    Returns a list of Documents (one per page for PDFs).
    """
    suffix = path.suffix.lower()
    base_meta = {"file_path": str(path), "file_name": path.name}
    docs: list[Document] = []

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                docs.append(Document(
                    text=text,
                    metadata={**base_meta, "page_label": str(i + 1)},
                    id_=f"{path.name}::page{i+1}",
                ))

    elif suffix == ".docx":
        import docx2txt
        text = (docx2txt.process(str(path)) or "").strip()
        if text:
            docs.append(Document(text=text, metadata=base_meta, id_=path.name))

    elif suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append(Document(text=text, metadata=base_meta, id_=path.name))

    return docs


# ── Singletons ────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def _get_embed_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=config.EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def _get_llm() -> OpenAILike:
    return OpenAILike(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        api_base=GROQ_API_BASE,
        is_chat_model=True,
        context_window=128_000,
        max_tokens=1024,
    )


@st.cache_resource(show_spinner=False)
def _get_chroma_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    return client, client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _configure_settings() -> None:
    Settings.embed_model = _get_embed_model()
    Settings.llm = _get_llm()
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP
    Settings.num_output = 1024


# ── Index ─────────────────────────────────────────────────

def get_index() -> VectorStoreIndex:
    _configure_settings()
    _, collection = _get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )


def ingest_documents(file_paths: list[Path]) -> tuple[int, list[str]]:
    """
    Extract text from documents and index them.
    Skips files already indexed (by content hash).
    Returns (chunks_added, skipped_filenames).
    """
    _configure_settings()
    _, collection = _get_chroma_collection()

    # Collect existing file hashes to avoid re-indexing
    existing_hashes: set[str] = set()
    if collection.count() > 0:
        for meta in collection.get(include=["metadatas"]).get("metadatas", []):
            if meta and "file_hash" in meta:
                existing_hashes.add(meta["file_hash"])

    new_paths, skipped = [], []
    for fp in file_paths:
        if _file_hash(fp) in existing_hashes:
            skipped.append(fp.name)
        else:
            new_paths.append(fp)

    if not new_paths:
        return 0, skipped

    # Load text from each file
    all_docs: list[Document] = []
    failed: list[str] = []
    for fp in new_paths:
        try:
            docs = _load_file(fp)
            if docs:
                fhash = _file_hash(fp)
                for doc in docs:
                    doc.metadata["file_hash"] = fhash
                all_docs.extend(docs)
            else:
                failed.append(f"{fp.name} (no text extracted)")
        except Exception as exc:
            failed.append(f"{fp.name} ({exc})")

    if failed:
        st.warning(f"Could not extract text from: {', '.join(failed)}")

    if not all_docs:
        return 0, skipped

    # Split into chunks
    nodes = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    ).get_nodes_from_documents(all_docs)

    # Index into ChromaDB
    _, collection = _get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    VectorStoreIndex(
        nodes,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        show_progress=True,
    )
    return len(nodes), skipped


def query_with_sources(question: str, top_k: int = config.TOP_K) -> dict[str, Any]:
    engine = get_index().as_query_engine(
        similarity_top_k=top_k,
        response_mode="compact",
    )
    response = engine.query(question)

    sources = []
    for node_score in response.source_nodes:
        node: NodeWithScore = node_score
        sources.append({
            "text": node.node.get_content()[:600],
            "score": round(float(node.score or 0), 4),
            "file": Path(node.node.metadata.get("file_path", "unknown")).name,
            "page": node.node.metadata.get("page_label", "—"),
        })

    return {"answer": str(response), "sources": sources, "raw_response": response}


def get_collection_stats() -> dict[str, Any]:
    _, collection = _get_chroma_collection()
    count = collection.count()
    files: set[str] = set()
    if count > 0:
        for meta in collection.get(include=["metadatas"]).get("metadatas", []):
            if meta and meta.get("file_path"):
                files.add(Path(meta["file_path"]).name)
    return {"total_chunks": count, "unique_files": sorted(files)}


def delete_collection() -> None:
    client, _ = _get_chroma_collection()
    client.delete_collection(config.COLLECTION_NAME)
    client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    _get_chroma_collection.clear()


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
