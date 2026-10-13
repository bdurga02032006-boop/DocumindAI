"""Tests for PDF loading without external API calls."""

from io import BytesIO

import pytest
from pypdf import PdfWriter

from services.document_loader import (
    DocumentProcessingError,
    load_and_split_pdf,
    save_uploaded_pdf,
)


def make_pdf(*pages: str) -> bytes:
    """Create a small valid PDF fixture with blank pages."""
    writer = PdfWriter()
    for _ in pages:
        writer.add_blank_page(width=300, height=300)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_save_uploaded_pdf_rejects_non_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr("services.document_loader.UPLOADS_DIR", tmp_path)
    with pytest.raises(DocumentProcessingError, match="PDF"):
        save_uploaded_pdf(BytesIO(b"not a pdf"), "notes.txt")


def test_load_and_split_pdf_rejects_invalid_pdf(tmp_path):
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_bytes(b"not a pdf")
    with pytest.raises(DocumentProcessingError, match="could not be read"):
        load_and_split_pdf(invalid_pdf)


def test_load_and_split_pdf_rejects_empty_text_pdf(tmp_path):
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(make_pdf(""))
    with pytest.raises(DocumentProcessingError, match="readable text"):
        load_and_split_pdf(empty_pdf)


def test_load_and_split_pdf_preserves_page_metadata(tmp_path, monkeypatch):
    source = tmp_path / "handbook.pdf"
    source.write_bytes(b"fixture")

    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        pages = [FakePage("Page one content"), FakePage("Page two content")]

    monkeypatch.setattr("services.document_loader.PdfReader", lambda _: FakeReader())
    chunks, document_id = load_and_split_pdf(source)

    assert len(chunks) == 2
    assert {chunk.metadata["page"] for chunk in chunks} == {1, 2}
    assert all(chunk.metadata["filename"] == "handbook.pdf" for chunk in chunks)
    assert all(chunk.metadata["document_id"] == document_id for chunk in chunks)
