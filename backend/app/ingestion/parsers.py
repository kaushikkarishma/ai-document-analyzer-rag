from pathlib import Path

import pdfplumber
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def validate_supported_file(path: Path) -> None:
    """Reject file types we do not know how to parse yet.

    A RAG pipeline is only as good as the text it receives. Being strict here
    prevents us from silently accepting unsupported files and producing empty
    or misleading chunks later.
    """

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Supported types: {supported}")


def parse_document(path: Path) -> list[dict]:
    """Parse a supported document into page-aware text records.

    The return value is a list of dictionaries instead of one giant string
    because page metadata becomes important later when we cite sources in RAG
    answers. TXT and DOCX files do not have reliable page numbers, so we mark
    their page as 1.
    """

    validate_supported_file(path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        return _parse_pdf(path)
    if extension == ".txt":
        return _parse_txt(path)
    if extension == ".docx":
        return _parse_docx(path)

    raise ValueError(f"Unsupported file type: {extension}")


def _parse_pdf(path: Path) -> list[dict]:
    """Extract text page by page from a PDF using pdfplumber."""

    pages: list[dict] = []

    with pdfplumber.open(path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            # PDF text extraction can return None for image-only or blank pages.
            # We normalize that to an empty string so the rest of the pipeline
            # can handle all pages consistently.
            text = page.extract_text() or ""
            if text.strip():
                pages.append(
                    {
                        "page_number": page_index,
                        "text": _normalize_text(text),
                    }
                )

    return pages


def _parse_txt(path: Path) -> list[dict]:
    """Read plain text with UTF-8 first, then fall back for older files."""

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Some Windows-created text files use a legacy encoding. This fallback
        # keeps the app usable without turning parsing into an encoding lesson.
        text = path.read_text(encoding="latin-1")

    return [{"page_number": 1, "text": _normalize_text(text)}] if text.strip() else []


def _parse_docx(path: Path) -> list[dict]:
    """Extract paragraph text from a Word document."""

    document = Document(path)
    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]
    text = "\n\n".join(paragraphs)

    return [{"page_number": 1, "text": _normalize_text(text)}] if text.strip() else []


def _normalize_text(text: str) -> str:
    """Do light cleanup while preserving paragraph boundaries.

    We avoid aggressive cleanup because document structure helps chunking. For
    example, paragraph breaks are useful natural split points.
    """

    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
