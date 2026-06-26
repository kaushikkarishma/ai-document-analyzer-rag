# AI Document Analyzer

A local Document Intelligence Assistant built in phases to learn the full RAG
pipeline: parsing, chunking, embeddings, vector search, prompt construction,
and local generation through Ollama.

## Phase 0 Architecture

```text
Upload
  -> Parse
  -> Chunk
  -> Embed
  -> Store
  -> Query
  -> Retrieve
  -> Augment Prompt
  -> Ollama
  -> Response with Sources
```

## Current Stack

- Backend: FastAPI
- Local LLM: Ollama with `llama3.1:8b`
- Local embeddings: Ollama with `nomic-embed-text`
- Vector store: ChromaDB
- Document parsing: `pdfplumber` and `python-docx`
- Frontend: React with Vite

## Phase 0 Setup

From the `backend` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open:

- API health: <http://127.0.0.1:8000/health>
- Ollama health: <http://127.0.0.1:8000/health/ollama>
- API docs: <http://127.0.0.1:8000/docs>

## Phase 1 Endpoint

Use FastAPI docs to test document parsing and chunking:

1. Open <http://127.0.0.1:8000/docs>
2. Expand `POST /documents/preview`
3. Upload a `.pdf`, `.txt`, or `.docx`
4. Inspect the returned pages and chunks before moving on to embeddings

## Phase 2 Endpoints

Use these endpoints to test embeddings and vector search without involving the
LLM answer generator yet:

1. Open <http://127.0.0.1:8000/docs>
2. Expand `POST /documents/index`
3. Upload a `.pdf`, `.txt`, or `.docx`
4. Confirm the response reports the document's `chunk_count`
5. Expand `POST /search`
6. Send a question such as:

```json
{
  "query": "What is this assignment about?",
  "top_k": 4
}
```

The response should show the most similar chunks, their distances, and source
metadata such as file name and page number.

Make sure these Ollama models are available:

```powershell
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```
