"""Retrieval-augmented answer generation."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from services.vector_store import get_retriever

FALLBACK_ANSWER = "I couldn't find that information in the uploaded documents."


class RAGError(Exception):
    """Raised when retrieval or answer generation fails."""


def answer_question(store, question: str) -> tuple[str, list[Document]]:
    """Retrieve relevant chunks and answer only from their contents."""
    try:
        documents = get_retriever(store).invoke(question)
        if not documents:
            return FALLBACK_ANSWER, []
        context = "\n\n".join(
            f"Source: {document.metadata.get('filename', 'document')} - "
            f"Page {document.metadata.get('page', '?')}\n{document.page_content}"
            for document in documents
        )
        prompt = (
            "You answer questions about uploaded documents. Use only the context "
            "below. If the answer is not explicitly supported by the context, reply "
            f"exactly: {FALLBACK_ANSWER}\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        response = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        ).invoke(prompt)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        return answer.strip() or FALLBACK_ANSWER, documents
    except RAGError:
        raise
    except Exception as exc:
        error_text = str(exc).lower()
        if "connection" in error_text or "refused" in error_text or "11434" in error_text:
            message = (
                "Ollama is not running. Install Ollama, start it, and try again."
            )
        elif "not found" in error_text or "model" in error_text and "pull" in error_text:
            message = f"Ollama model '{OLLAMA_MODEL}' is missing. Run: ollama pull {OLLAMA_MODEL}"
        else:
            message = "The local Ollama model could not answer. Check Ollama and try again."
        raise RAGError(message) from exc
