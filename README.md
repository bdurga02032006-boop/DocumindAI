# DocuMind AI

DocuMind AI is a fully local Streamlit document intelligence chatbot. Upload a text-based PDF, index it with a local Sentence Transformers model and ChromaDB, then ask questions answered by a local Ollama model.

## Features

- PDF upload with page-aware text extraction
- Recursive text chunking with source metadata
- Persistent ChromaDB vector storage
- Local Sentence Transformers embeddings
- Local Ollama chat completion
- Retrieval-grounded answers with source pages
- Duplicate indexing avoidance
- Session-scoped chat history
- Friendly handling for invalid PDFs, missing configuration, and service failures

## Architecture

```text
PDF upload -> PyPDF extraction -> LangChain chunking -> local embeddings
          -> persistent ChromaDB -> similarity retrieval -> local Ollama answer
```

## Tech Stack

Python 3.10.1, Streamlit, LangChain, Sentence Transformers, Ollama, ChromaDB, PyPDF, python-dotenv, and pytest.

## Folder Structure

```text
app.py
config/
  settings.py
services/
  document_loader.py
  embeddings.py
  vector_store.py
  rag_chain.py
data/
  uploads/
  chroma/
tests/
  test_document_loader.py
```

Runtime data directories are created automatically and are ignored by Git.

## Installation

Create and activate a Python 3.10 virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

## Environment Setup

Copy `.env.example` to `.env` if you want to customize the local Ollama connection:

```text
OLLAMA_MODEL=gemma3:4b
OLLAMA_BASE_URL=http://localhost:11434
```

No API key or paid service is required. The embedding model downloads from Hugging Face the first time it is used and then runs locally.

## Ollama Setup

Install Ollama for Windows from [ollama.com/download](https://ollama.com/download/windows), then open PowerShell and run:

```powershell
ollama serve
```

In a second PowerShell window, download the default local answer model:

```powershell
ollama pull gemma3:4b
```

Confirm the model is available:

```powershell
ollama list
```

## Run

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit, upload a readable PDF, select **Process document**, and ask a question in the chat box.

## How RAG Works

Each PDF page is extracted and split into overlapping chunks. Chunks are embedded by the local Sentence Transformers model and stored persistently in ChromaDB. For each question, the nearest chunks are retrieved and passed to Ollama as context. The prompt requires the local model to use only that context and return a clear fallback when the answer is not supported.

## Example Usage

Upload an employee handbook and ask: `How many days of annual leave do employees receive?` The response includes the answer and the PDF page used as its source.

## Testing

The document loader tests create local PDF fixtures and do not require Ollama, an API key, or network access:

```powershell
pytest
```

## Future Improvements

- Support additional document formats
- Add document removal and collection management
- Add streaming answers and richer citation links
- Add optional per-user document workspaces
