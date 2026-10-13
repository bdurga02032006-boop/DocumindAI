"""Persistent Chroma indexing and retrieval."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import CHROMA_DIR, RETRIEVAL_K
from services.embeddings import get_embeddings

# Keep the current local embedding dimensions isolated from older collections.
COLLECTION_NAME = "documind_local_documents"


def get_vector_store() -> Chroma:
    """Open the persistent Chroma collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(Path(CHROMA_DIR)),
        embedding_function=get_embeddings(),
    )


def index_documents(documents: list[Document], document_id: str) -> tuple[Chroma, bool]:
    """Index a document once and return the store plus whether it was new."""
    store = get_vector_store()
    existing = store.get(where={"document_id": document_id}, limit=1)
    if existing.get("ids"):
        return store, False
    store.add_documents(documents)
    return store, True


def get_retriever(store: Chroma):
    """Create a similarity retriever for the indexed collection."""
    return store.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVAL_K})
