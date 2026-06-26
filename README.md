# AI Document Analyzer (RAG + Ollama)

A Retrieval-Augmented Generation (RAG) based document question-answering application that allows users to upload documents, index them using vector embeddings, and ask natural language questions using a locally running LLM via Ollama.

---

## Features

- Upload PDF and text documents
- Automatic document parsing and chunking
- Vector embeddings using local embedding models
- ChromaDB vector storage
- Semantic similarity search
- Context-aware answer generation using Ollama
- React frontend with FastAPI backend
- Fully local inference (no cloud APIs required)

---

## Tech Stack

### Backend
- FastAPI
- Python
- ChromaDB
- Ollama
- HTTPX

### Frontend
- React
- Vite
- JavaScript

### AI / ML
- Retrieval-Augmented Generation (RAG)
- Vector Embeddings
- Semantic Search
- Prompt Engineering

---

## Project Structure

```
backend/
    app/
        ingestion/
        rag/
        main.py

frontend/
    src/
    public/

README.md
```

---

## RAG Pipeline

```
Document Upload
        │
        ▼
Document Parsing
        │
        ▼
Chunking
        │
        ▼
Embedding Generation
        │
        ▼
ChromaDB Vector Store
        │
        ▼
Similarity Search
        │
        ▼
Prompt Construction
        │
        ▼
Ollama
        │
        ▼
Generated Answer
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/kaushikkarishma/ai-document-analyzer-rag.git
```

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Future Improvements

- Multi-document support
- Conversation history
- Streaming responses
- User authentication
- Hybrid search
- OCR support

---

## Author

Karishma Kaushik

B.Tech AIML | COER University

Learning Retrieval-Augmented Generation, LLMs and AI Application Development.