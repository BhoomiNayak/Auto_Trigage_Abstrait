"""Text extraction service for PDF and plain text documents."""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


class ExtractionError(Exception):
    """Raised when text extraction fails."""

    pass


@dataclass
class ExtractionMetadata:
    """Metadata about the extracted text."""

    filename: str
    file_type: str
    page_count: int
    char_count: int


@dataclass
class ExtractedDocument:
    """Result of text extraction from a file."""

    text: str
    metadata: ExtractionMetadata


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".text"}


def get_file_extension(filename: str) -> str:
    """Get the lowercase file extension."""
    return Path(filename).suffix.lower()


def is_supported_file(filename: str) -> bool:
    """Check if the file type is supported for extraction."""
    return get_file_extension(filename) in SUPPORTED_EXTENSIONS


def extract_from_pdf(content: bytes, filename: str) -> ExtractedDocument:
    """Extract text from a PDF file.

    Args:
        content: Raw PDF file bytes.
        filename: Original filename for metadata.

    Returns:
        ExtractedDocument with text and metadata.

    Raises:
        ExtractionError: If the PDF cannot be read or contains no text.
    """
    # Validate PDF magic bytes (files should start with %PDF)
    if not content[:5].startswith(b"%PDF"):
        raise ExtractionError(
            "File does not appear to be a valid PDF (invalid header). "
            "Please ensure the file is not corrupted."
        )

    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        raise ExtractionError(f"Failed to open PDF: {e}")

    pages_text = []
    for page in doc:
        page_text = page.get_text()
        if page_text.strip():
            pages_text.append(page_text)

    doc.close()

    if not pages_text:
        raise ExtractionError(
            "PDF contains no extractable text. "
            "It may be a scanned document (OCR not supported in this version)."
        )

    full_text = "\n\n".join(pages_text)

    return ExtractedDocument(
        text=full_text,
        metadata=ExtractionMetadata(
            filename=filename,
            file_type="pdf",
            page_count=len(pages_text),
            char_count=len(full_text),
        ),
    )


def extract_from_text(content: bytes, filename: str) -> ExtractedDocument:
    """Extract text from a plain text file.

    Args:
        content: Raw text file bytes.
        filename: Original filename for metadata.

    Returns:
        ExtractedDocument with text and metadata.

    Raises:
        ExtractionError: If the file cannot be decoded or is empty.
    """
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except UnicodeDecodeError:
            raise ExtractionError("Failed to decode text file. Unsupported encoding.")

    if not text.strip():
        raise ExtractionError("Document is empty — no text content found.")

    return ExtractedDocument(
        text=text,
        metadata=ExtractionMetadata(
            filename=filename,
            file_type="text",
            page_count=1,
            char_count=len(text),
        ),
    )


def extract_text(content: bytes, filename: str) -> ExtractedDocument:
    """Extract text from a file based on its extension.

    This is the main entry point for the extraction service.

    Args:
        content: Raw file bytes.
        filename: Original filename (used to determine file type).

    Returns:
        ExtractedDocument with extracted text and metadata.

    Raises:
        ExtractionError: If file type is unsupported or extraction fails.
    """
    extension = get_file_extension(filename)

    if not is_supported_file(filename):
        raise ExtractionError(
            f"Unsupported file type '{extension}'. Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if extension == ".pdf":
        return extract_from_pdf(content, filename)
    else:
        # .txt, .md, .text — all treated as plain text
        return extract_from_text(content, filename)
