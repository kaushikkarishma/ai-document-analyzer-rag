import httpx

from app.config import settings


class EmbeddingError(RuntimeError):
    """Raised when Ollama cannot create embeddings for our text."""


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert text chunks into embedding vectors through Ollama.

    Ollama's `/api/embed` endpoint takes text and returns arrays of numbers.
    Those numbers are not human-readable, but distance between vectors is
    meaningful: chunks about similar topics should produce vectors that are
    close together.
    """

    if not texts:
        return []

    payload = {
        "model": settings.embedding_model,
        "input": texts,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/embed",
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise EmbeddingError(
            "Could not create embeddings. Make sure Ollama is running and "
            f"the `{settings.embedding_model}` model is pulled."
        ) from exc

    data = response.json()
    embeddings = data.get("embeddings")

    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise EmbeddingError("Ollama returned an unexpected embedding response.")

    return embeddings


async def embed_query(query: str) -> list[float]:
    """Create one embedding vector for a user's search question."""

    embeddings = await embed_texts([query])
    return embeddings[0]
