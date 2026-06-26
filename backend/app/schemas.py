from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response returned by the basic backend health endpoint."""

    status: str
    app_name: str


class OllamaHealthResponse(BaseModel):
    """Response returned after checking whether the Ollama server is reachable."""

    status: str
    base_url: str
    chat_model: str
    embedding_model: str


class ParsedPage(BaseModel):
    """One parsed page or page-like text section from an uploaded document."""

    page_number: int
    text_preview: str
    char_count: int


class ChunkPreview(BaseModel):
    """A chunk preview returned before we add embeddings in Phase 2."""

    chunk_id: int
    source_file: str
    page_number: int
    char_count: int
    text: str


class DocumentPreviewResponse(BaseModel):
    """Response for the Phase 1 upload-and-chunk endpoint."""

    filename: str
    saved_path: str
    page_count: int
    chunk_count: int
    pages: list[ParsedPage]
    chunks: list[ChunkPreview]


class DocumentIndexResponse(BaseModel):
    """Response returned after chunks are embedded and stored."""

    filename: str
    page_count: int
    chunk_count: int
    stored_chunk_count: int


class SearchRequest(BaseModel):
    """Request body for raw vector similarity search."""

    query: str
    top_k: int = 4


class SearchResult(BaseModel):
    """One retrieved chunk from ChromaDB."""

    id: str
    source_file: str
    page_number: int
    chunk_id: int
    distance: float
    text: str


class SearchResponse(BaseModel):
    """Raw retrieval response before LLM generation is introduced."""

    query: str
    top_k: int
    results: list[SearchResult]


class AskRequest(BaseModel):
    """Request body for the final RAG question-answering endpoint."""

    question: str
    top_k: int = 4


class SourceCitation(BaseModel):
    """Source information shown beside generated answers."""

    source_number: int
    source_file: str
    page_number: int
    chunk_id: int
    distance: float
    text_preview: str


class AskResponse(BaseModel):
    """Final RAG response: answer plus citations."""

    question: str
    answer: str
    sources: list[SourceCitation]


class DocumentListItem(BaseModel):
    """A document file currently saved in the upload folder."""

    filename: str
    size_bytes: int


class DocumentListResponse(BaseModel):
    """Response for listing uploaded documents."""

    documents: list[DocumentListItem]
    stored_chunk_count: int
