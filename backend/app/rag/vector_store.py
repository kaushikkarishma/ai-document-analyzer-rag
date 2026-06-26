from app.config import settings
from app.ingestion.chunking import TextChunk

import chromadb


COLLECTION_NAME = "document_chunks"


def get_collection():
    """Open the local persistent ChromaDB collection.

    ChromaDB stores vectors on disk in `backend/data/chroma`, so indexed chunks
    survive a server restart. We use cosine distance because embeddings are
    usually compared by direction in vector space, not raw magnitude.
    """

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(chunks: list[TextChunk], embeddings: list[list[float]]) -> int:
    """Store chunk text, metadata, and embeddings in ChromaDB."""

    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have exactly one embedding.")
    if not chunks:
        return 0

    collection = get_collection()

    ids = [
        f"{chunk.source_file}:page-{chunk.page_number}:chunk-{chunk.chunk_id}"
        for chunk in chunks
    ]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "source_file": chunk.source_file,
            "page_number": chunk.page_number,
            "chunk_id": chunk.chunk_id,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        }
        for chunk in chunks
    ]

    # Upsert lets us re-index the same file during learning without manually
    # clearing the database each time. Matching ids are replaced.
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(chunks)


def similarity_search(query_embedding: list[float], top_k: int = 4) -> list[dict]:
    """Return the closest stored chunks for a query embedding."""

    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    matches: list[dict] = []
    for result_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        matches.append(
            {
                "id": result_id,
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return matches


def count_chunks() -> int:
    """Return how many chunk vectors are currently stored."""

    return get_collection().count()
