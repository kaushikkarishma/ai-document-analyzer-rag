from pathlib import Path
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import httpx

from app.config import settings
from app.ingestion.chunking import chunk_pages
from app.ingestion.parsers import SUPPORTED_EXTENSIONS, parse_document
from app.rag.embeddings import EmbeddingError, embed_query, embed_texts
from app.rag.generation import GenerationError, generate_answer
from app.rag.vector_store import count_chunks, similarity_search, upsert_chunks
from app.schemas import (
    AskRequest,
    AskResponse,
    ChunkPreview,
    DocumentListItem,
    DocumentListResponse,
    DocumentIndexResponse,
    DocumentPreviewResponse,
    HealthResponse,
    OllamaHealthResponse,
    ParsedPage,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SourceCitation,
)


app = FastAPI(
    title=settings.app_name,
    description="A local RAG app for asking questions over uploaded documents.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Confirm the FastAPI server itself is running.

    This deliberately does not check Ollama or ChromaDB. A narrow health check
    makes debugging easier: first prove the API process is alive, then check
    external dependencies one by one.
    """

    return HealthResponse(status="ok", app_name=settings.app_name)


@app.get("/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    """List uploaded documents and the current vector-store chunk count."""

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    documents = [
        DocumentListItem(filename=path.name, size_bytes=path.stat().st_size)
        for path in sorted(settings.upload_dir.iterdir())
        if path.is_file() and path.name != ".gitkeep"
    ]

    return DocumentListResponse(
        documents=documents,
        stored_chunk_count=count_chunks(),
    )


@app.get("/health/ollama", response_model=OllamaHealthResponse)
async def ollama_health() -> OllamaHealthResponse:
    """Confirm the backend can reach the local Ollama server.

    Ollama exposes a local HTTP API on port 11434 by default. We call /api/tags
    because it is a lightweight endpoint that lists locally available models
    without asking the LLM to generate text.
    """

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is not reachable. Make sure the Ollama app is running "
                "and that you pulled the required models."
            ),
        ) from exc

    return OllamaHealthResponse(
        status="ok",
        base_url=settings.ollama_base_url,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
    )


@app.post("/documents/preview", response_model=DocumentPreviewResponse)
async def preview_document(file: UploadFile = File(...)) -> DocumentPreviewResponse:
    """Upload a document, parse its text, and return chunk previews.

    This endpoint is intentionally not storing embeddings yet. In Phase 1, the
    learning goal is to inspect whether parsing and chunking are working before
    we add vector search. That separation makes bugs much easier to diagnose.
    """

    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Please upload one of: {supported}",
        )

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = settings.upload_dir / original_name

    with saved_path.open("wb") as output_file:
        shutil.copyfileobj(file.file, output_file)

    try:
        pages = parse_document(saved_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No readable text found in this document.",
        )

    chunks = chunk_pages(pages, source_file=original_name)

    return DocumentPreviewResponse(
        filename=original_name,
        saved_path=str(saved_path),
        page_count=len(pages),
        chunk_count=len(chunks),
        pages=[
            ParsedPage(
                page_number=page["page_number"],
                text_preview=page["text"][:500],
                char_count=len(page["text"]),
            )
            for page in pages
        ],
        chunks=[
            ChunkPreview(
                chunk_id=chunk.chunk_id,
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                char_count=len(chunk.text),
                text=chunk.text,
            )
            for chunk in chunks
        ],
    )


@app.post("/documents/index", response_model=DocumentIndexResponse)
async def index_document(file: UploadFile = File(...)) -> DocumentIndexResponse:
    """Upload, parse, chunk, embed, and store a document in ChromaDB.

    This endpoint is the Phase 2 version of `/documents/preview`. The important
    new step is embedding: we convert each chunk into a vector through Ollama,
    then store those vectors in ChromaDB so they can be searched later.
    """

    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Please upload one of: {supported}",
        )

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = settings.upload_dir / original_name

    with saved_path.open("wb") as output_file:
        shutil.copyfileobj(file.file, output_file)

    try:
        pages = parse_document(saved_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No readable text found in this document.",
        )

    chunks = chunk_pages(pages, source_file=original_name)

    try:
        embeddings = await embed_texts([chunk.text for chunk in chunks])
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    upsert_chunks(chunks, embeddings)

    return DocumentIndexResponse(
        filename=original_name,
        page_count=len(pages),
        chunk_count=len(chunks),
        stored_chunk_count=count_chunks(),
    )


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest) -> SearchResponse:
    """Search stored chunks by semantic similarity, without using the LLM yet."""

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if request.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be greater than 0.")

    try:
        query_embedding = await embed_query(query)
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    matches = similarity_search(query_embedding, top_k=request.top_k)

    return SearchResponse(
        query=query,
        top_k=request.top_k,
        results=[
            SearchResult(
                id=match["id"],
                source_file=match["metadata"]["source_file"],
                page_number=match["metadata"]["page_number"],
                chunk_id=match["metadata"]["chunk_id"],
                distance=match["distance"],
                text=match["text"],
            )
            for match in matches
        ],
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """Run the full RAG flow: embed query, retrieve chunks, ask Ollama."""

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if request.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be greater than 0.")

    try:
        query_embedding = await embed_query(question)
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    matches = similarity_search(query_embedding, top_k=request.top_k)
    if not matches:
        raise HTTPException(
            status_code=404,
            detail="No indexed chunks found. Upload and index a document first.",
        )

    try:
        answer = await generate_answer(question, matches)
    except GenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AskResponse(
        question=question,
        answer=answer,
        sources=[
            SourceCitation(
                source_number=index,
                source_file=match["metadata"]["source_file"],
                page_number=match["metadata"]["page_number"],
                chunk_id=match["metadata"]["chunk_id"],
                distance=match["distance"],
                text_preview=match["text"][:500],
            )
            for index, match in enumerate(matches, start=1)
        ],
    )
