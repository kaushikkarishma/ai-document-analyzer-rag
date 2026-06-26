import httpx

from app.config import settings


class GenerationError(RuntimeError):
    """Raised when the local Ollama chat model cannot generate an answer."""


def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> list[dict]:
    """Build the messages sent to the local LLM.

    The system message sets the behavior rules. The user message contains both
    the question and retrieved document chunks. This is the core RAG pattern:
    retrieve relevant context first, then ask the model to answer only from that
    context.
    """

    context_blocks = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]
        context_blocks.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"File: {metadata['source_file']}",
                    f"Page: {metadata['page_number']}",
                    f"Chunk: {metadata['chunk_id']}",
                    chunk["text"],
                ]
            )
        )

    context = "\n\n---\n\n".join(context_blocks)

    return [
        {
            "role": "system",
            "content": (
                "You are a document intelligence assistant. Answer only using "
                "the provided context. If the context does not contain the "
                "answer, say you do not know from the uploaded documents. "
                "Cite sources inline like [Source 1] or [Source 2]."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Retrieved context:\n{context}\n\n"
                "Write a clear answer with source citations."
            ),
        },
    ]


async def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """Ask Ollama to generate an answer from retrieved chunks."""

    messages = build_rag_prompt(question, retrieved_chunks)
    payload = {
        "model": settings.chat_model,
        "messages": messages,
        "stream": False,
        "options": {
            # Lower temperature makes answers steadier and less creative,
            # which is what we want for grounded document Q&A.
            "temperature": 0.2,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise GenerationError(
            f"Ollama HTTP {exc.response.status_code}: {exc.response.text}"
        ) from exc

    except httpx.RequestError as exc:
        raise GenerationError(
            f"Could not connect to Ollama: {exc}"
        ) from exc

    data = response.json()
    message = data.get("message", {})
    content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        raise GenerationError("Ollama returned an empty answer.")

    return content.strip()
