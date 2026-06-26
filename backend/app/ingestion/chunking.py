from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """A chunk of text plus the metadata needed for source citation."""

    chunk_id: int
    text: str
    source_file: str
    page_number: int
    start_char: int
    end_char: int


def chunk_pages(
    pages: list[dict],
    source_file: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    """Split page text into overlapping chunks.

    Chunk overlap repeats the last part of one chunk at the start of the next
    chunk. This helps retrieval when an important idea crosses a chunk boundary:
    the model can still find enough surrounding context in either neighboring
    chunk.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[TextChunk] = []

    for page in pages:
        page_chunks = _recursive_split(page["text"], chunk_size)
        for page_chunk_index, page_chunk in enumerate(page_chunks):
            chunk_text = page_chunk.strip()
            if not chunk_text:
                continue

            if page_chunk_index > 0 and chunk_overlap:
                previous_tail = page_chunks[page_chunk_index - 1][-chunk_overlap:].strip()
                if previous_tail:
                    chunk_text = f"{previous_tail}\n{chunk_text}"

            chunks.append(
                TextChunk(
                    chunk_id=len(chunks) + 1,
                    text=chunk_text,
                    source_file=source_file,
                    page_number=page["page_number"],
                    start_char=page["text"].find(page_chunk),
                    end_char=page["text"].find(page_chunk) + len(page_chunk),
                )
            )

    return chunks


def _recursive_split(text: str, chunk_size: int) -> list[str]:
    """Split text by increasingly smaller natural boundaries.

    The idea is simple: prefer paragraph-sized chunks, then line-sized chunks,
    then sentence/word boundaries. Only if the text is still too large do we use
    a hard character cut. This approximates what libraries like LangChain's
    RecursiveCharacterTextSplitter do, but it stays readable for learning.
    """

    separators = ["\n\n", "\n", ". ", " ", ""]
    return _split_with_separators(text, chunk_size, separators)


def _split_with_separators(
    text: str,
    chunk_size: int,
    separators: list[str],
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    separator = separators[0]
    remaining_separators = separators[1:]

    if separator == "":
        return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]

    pieces = text.split(separator)
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        candidate = piece if not current else f"{current}{separator}{piece}"

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.extend(_split_with_separators(current, chunk_size, remaining_separators))

        current = piece

    if current:
        chunks.extend(_split_with_separators(current, chunk_size, remaining_separators))

    return chunks
