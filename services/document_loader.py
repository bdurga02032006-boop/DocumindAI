"""PDF validation, storage, extraction, and chunking."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, UPLOADS_DIR


class DocumentProcessingError(Exception):
    """Raised when an uploaded PDF cannot be processed."""


def save_uploaded_pdf(uploaded_file: BinaryIO, filename: str) -> Path:
    """Save an uploaded PDF below the controlled uploads directory."""
    if not filename.lower().endswith(".pdf"):
        raise DocumentProcessingError("Please upload a PDF file.")
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    destination = UPLOADS_DIR / safe_name
    destination.write_bytes(uploaded_file.getvalue())
    if destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise DocumentProcessingError("The uploaded PDF is empty.")
    return destination


def load_and_split_pdf(file_path: str | Path) -> tuple[list[Document], str]:
    """Extract every PDF page and split it into retrieval-sized documents."""
    path = Path(file_path)
    try:
        reader = PdfReader(str(path))
        if not reader.pages:
            raise DocumentProcessingError("The PDF contains no pages.")
        page_documents: list[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                page_documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "filename": path.name,
                            "page": page_number,
                        },
                    )
                )
        if not page_documents:
            raise DocumentProcessingError("No readable text was found in the PDF.")
    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError("The PDF could not be read.") from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(page_documents)
    document_id = hashlib.sha256(path.read_bytes()).hexdigest()
    for chunk in chunks:
        chunk.metadata["document_id"] = document_id
    return chunks, document_id
