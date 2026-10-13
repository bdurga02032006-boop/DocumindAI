"""Local Sentence Transformers embedding configuration."""

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import EMBEDDING_MODEL


class LocalEmbeddingError(Exception):
    """Raised when the local embedding model cannot be loaded."""


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create embeddings using a local Sentence Transformers model."""
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise LocalEmbeddingError(
            "The local embedding model could not be loaded. "
            "Check your internet connection for the first model download."
        ) from exc
