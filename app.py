"""DocuMind AI Streamlit application."""

from __future__ import annotations

import streamlit as st

from config.settings import OLLAMA_MODEL, ensure_data_directories
from services.document_loader import (
	DocumentProcessingError,
	load_and_split_pdf,
	save_uploaded_pdf,
)
from services.rag_chain import RAGError, answer_question
from services.embeddings import LocalEmbeddingError
from services.vector_store import index_documents


def initialize_session() -> None:
	"""Initialize state that should survive Streamlit reruns."""
	if "messages" not in st.session_state:
		st.session_state.messages = []
	if "vector_store" not in st.session_state:
		st.session_state.vector_store = None
	if "document_name" not in st.session_state:
		st.session_state.document_name = None
	if "processing_status" not in st.session_state:
		st.session_state.processing_status = "No document processed yet."


def render_sources(documents) -> None:
	"""Render unique source filename and page references."""
	sources = {
		(document.metadata.get("filename", "document"), document.metadata.get("page", "?"))
		for document in documents
	}
	if sources:
		st.markdown("**Sources**")
		for filename, page in sorted(sources):
			st.markdown(f"- {filename} - Page {page}")


def process_document(uploaded_file) -> None:
	"""Save, extract, embed, and index the selected PDF."""
	try:
		with st.spinner("Reading and indexing your document..."):
			file_path = save_uploaded_pdf(uploaded_file, uploaded_file.name)
			chunks, document_id = load_and_split_pdf(file_path)
			store, was_indexed = index_documents(chunks, document_id)
		st.session_state.vector_store = store
		st.session_state.document_name = uploaded_file.name
		st.session_state.processing_status = (
			"Document ready for questions."
			if was_indexed
			else "Document already indexed and ready for questions."
		)
		st.success(st.session_state.processing_status)
	except DocumentProcessingError as exc:
		st.error(str(exc))
		st.session_state.processing_status = "Document processing failed."
	except LocalEmbeddingError as exc:
		st.error(str(exc))
		st.session_state.processing_status = "Local embedding model failed."
	except Exception:
		st.error("Local document indexing failed. Check the embedding model and try again.")
		st.session_state.processing_status = "Document processing failed."


def main() -> None:
	"""Render the DocuMind AI application."""
	st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")
	ensure_data_directories()
	initialize_session()

	st.title("DocuMind AI")
	st.write("Ask focused questions about your uploaded PDF and get answers grounded in its content.")

	with st.sidebar:
		st.header("About DocuMind AI")
		st.caption("A document intelligence chatbot powered by retrieval-augmented generation.")
		st.divider()
		st.subheader("Document")
		if st.session_state.document_name:
			st.write(st.session_state.document_name)
		else:
			st.caption("No document loaded")
		st.subheader("Status")
		st.caption(st.session_state.processing_status)
		st.info(f"Local model: {OLLAMA_MODEL}")
		if st.button("Clear chat", use_container_width=True):
			st.session_state.messages = []
			st.rerun()

	uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
	if uploaded_file and st.button("Process document", type="primary"):
		process_document(uploaded_file)

	for message in st.session_state.messages:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
			if message.get("sources"):
				render_sources(message["sources"])

	question = st.chat_input("Ask a question about your document...")
	if question:
		if st.session_state.vector_store is None:
			st.warning("Please upload and process a PDF before asking a question.")
			return
		st.session_state.messages.append({"role": "user", "content": question})
		with st.chat_message("user"):
			st.markdown(question)
		with st.chat_message("assistant"):
			with st.spinner("Searching your document..."):
				try:
					answer, sources = answer_question(st.session_state.vector_store, question)
					st.markdown(answer)
					render_sources(sources)
					st.session_state.messages.append(
						{"role": "assistant", "content": answer, "sources": sources}
					)
				except RAGError as exc:
					st.error(str(exc))


if __name__ == "__main__":
	main()
