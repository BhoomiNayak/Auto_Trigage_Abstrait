"""FastAPI application for Auto-Triage & Document Extractor Agent."""

import os
import time
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models.base import DocumentType
from app.services.extractor import ExtractionError, extract_text, is_supported_file
from app.services.classifier import classify_document
from app.services.llm import LLMError, extract_document

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "An intelligent document processing agent that extracts structured data "
        "from invoices, support tickets, and resumes, with auto-triage tagging."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend/external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track server start time for uptime calculation
_start_time = time.time()


# --- Middleware for request ID and timing ---


@app.middleware("http")
async def add_headers_middleware(request: Request, call_next):
    """Add request ID and processing time headers to every response."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.time()

    response = await call_next(request)

    elapsed_ms = int((time.time() - start) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Processing-Time-Ms"] = str(elapsed_ms)

    return response


# --- Error Handlers ---


@app.exception_handler(ExtractionError)
async def extraction_error_handler(request: Request, exc: ExtractionError):
    """Handle document extraction errors."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "extraction_failed",
            "message": str(exc),
            "details": None,
        },
    )


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    """Handle LLM service errors."""
    error_msg = str(exc)
    if "rate_limit" in error_msg.lower() or "429" in error_msg:
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limited",
                "message": "LLM rate limit reached. Please retry.",
                "details": {"retry_after_seconds": 30},
            },
            headers={"Retry-After": "30"},
        )

    # Sanitize error — don't expose API keys or internal paths
    safe_message = f"LLM service error: {error_msg}"
    if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
        safe_message = "LLM authentication failed. Check your GEMINI_API_KEY configuration."
    elif "timeout" in error_msg.lower():
        safe_message = "LLM request timed out. The document may be too large."
    elif "not configured" in error_msg.lower():
        safe_message = "GEMINI_API_KEY is not configured. Set it in .env file."

    return JSONResponse(
        status_code=503,
        content={
            "error": "llm_unavailable",
            "message": safe_message,
            "details": None,
        },
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """Handle runtime errors (e.g., classifier failures)."""
    error_msg = str(exc)

    safe_message = "An internal error occurred. Please try again."
    if "not configured" in error_msg.lower():
        safe_message = "GEMINI_API_KEY is not configured. Set it in .env file."
    elif "classification failed" in error_msg.lower():
        safe_message = "Document classification service error. Please try again or provide doc_type."

    return JSONResponse(
        status_code=503,
        content={
            "error": "service_error",
            "message": safe_message,
            "details": None,
        },
    )


# --- Routes ---


@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup."""
    if not settings.is_configured:
        import warnings
        warnings.warn(
            "GEMINI_API_KEY is not configured! "
            "Set it in .env file or as environment variable. "
            "API calls will fail until configured.",
            stacklevel=2,
        )


@app.get("/api/v1/health")
async def health_check():
    """Service health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "llm_provider": "google_gemini",
        "model": settings.gemini_model,
        "uptime_seconds": int(time.time() - _start_time),
    }


@app.post("/api/v1/extract")
async def extract_from_document(
    file: UploadFile = File(..., description="PDF or TXT file to process"),
    doc_type: Optional[str] = Form(
        None,
        description="Document type hint: invoice, support_ticket, resume. Auto-detects if omitted.",
    ),
):
    """Extract structured data from a single document.

    Accepts a PDF or text file, extracts text, classifies (if needed),
    and returns structured JSON with auto-triage tags.
    """
    # Validate file type
    if not file.filename or not is_supported_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_file",
                "message": f"Unsupported file type. Accepted: .pdf, .txt, .md",
                "details": {"filename": file.filename},
            },
        )

    # Read file content
    content = await file.read()

    # Validate file size
    file_size_mb = len(content) / (1024 * 1024)
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "message": f"File exceeds {settings.max_file_size_mb}MB limit. Received: {file_size_mb:.1f}MB",
                "details": {
                    "max_size_mb": settings.max_file_size_mb,
                    "received_size_mb": round(file_size_mb, 1),
                },
            },
        )

    # Extract text from file
    extracted = extract_text(content, file.filename)

    # Determine document type
    if doc_type:
        # Validate provided doc_type
        try:
            document_type = DocumentType(doc_type)
        except ValueError:
            valid_types = [t.value for t in DocumentType if t != DocumentType.unknown]
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_doc_type",
                    "message": f"Invalid doc_type '{doc_type}'. Valid: {', '.join(valid_types)}",
                    "details": {"valid_types": valid_types},
                },
            )
    else:
        # Auto-classify
        classification = classify_document(extracted.text)
        document_type = classification.document_type

        if document_type == DocumentType.unknown:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "classification_failed",
                    "message": "Could not determine document type. Please provide doc_type parameter.",
                    "details": {
                        "classification_confidence": classification.confidence,
                        "reasoning": classification.reasoning,
                    },
                },
            )

    # Extract structured data
    result = extract_document(extracted.text, document_type)

    return result.model_dump()


@app.post("/api/v1/extract/batch")
async def extract_batch(
    files: list[UploadFile] = File(..., description="Multiple PDF or TXT files (max 10)"),
    doc_type: Optional[str] = Form(
        None,
        description="Apply same type to all files. Auto-detects per file if omitted.",
    ),
):
    """Extract structured data from multiple documents.

    Processes up to 10 files sequentially. Returns aggregated results
    with successful extractions and errors separated.
    """
    # Validate batch size
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "batch_too_large",
                "message": f"Maximum 10 files per batch. Received: {len(files)}",
                "details": {"max_files": 10, "received": len(files)},
            },
        )

    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "no_files",
                "message": "No files provided in batch request.",
            },
        )

    # Validate doc_type if provided
    document_type_hint = None
    if doc_type:
        try:
            document_type_hint = DocumentType(doc_type)
        except ValueError:
            valid_types = [t.value for t in DocumentType if t != DocumentType.unknown]
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_doc_type",
                    "message": f"Invalid doc_type '{doc_type}'. Valid: {', '.join(valid_types)}",
                    "details": {"valid_types": valid_types},
                },
            )

    results = []
    errors = []

    for file in files:
        filename = file.filename or "unknown"

        try:
            # Validate file type
            if not is_supported_file(filename):
                errors.append({
                    "filename": filename,
                    "error": "invalid_file",
                    "message": f"Unsupported file type. Accepted: .pdf, .txt, .md",
                })
                continue

            # Read content
            content = await file.read()

            # Validate size
            if len(content) > settings.max_file_size_bytes:
                errors.append({
                    "filename": filename,
                    "error": "file_too_large",
                    "message": f"File exceeds {settings.max_file_size_mb}MB limit.",
                })
                continue

            # Extract text
            extracted = extract_text(content, filename)

            # Determine type
            if document_type_hint:
                file_doc_type = document_type_hint
            else:
                classification = classify_document(extracted.text)
                file_doc_type = classification.document_type
                if file_doc_type == DocumentType.unknown:
                    errors.append({
                        "filename": filename,
                        "error": "classification_failed",
                        "message": f"Could not determine document type. Confidence: {classification.confidence}",
                    })
                    continue

            # Extract structured data
            result = extract_document(extracted.text, file_doc_type)
            result_dict = result.model_dump()
            result_dict["filename"] = filename
            results.append(result_dict)

        except ExtractionError as e:
            errors.append({
                "filename": filename,
                "error": "extraction_failed",
                "message": str(e),
            })
        except LLMError as e:
            errors.append({
                "filename": filename,
                "error": "llm_error",
                "message": str(e),
            })
        except Exception as e:
            errors.append({
                "filename": filename,
                "error": "internal_error",
                "message": f"Unexpected error: {str(e)}",
            })

    return {
        "total": len(files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }


@app.post("/api/v1/classify")
async def classify_only(
    file: UploadFile = File(..., description="PDF or TXT file to classify"),
):
    """Classify document type without full extraction.

    Useful for routing documents to different processing pipelines.
    """
    # Validate file type
    if not file.filename or not is_supported_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_file",
                "message": "Unsupported file type. Accepted: .pdf, .txt, .md",
                "details": {"filename": file.filename},
            },
        )

    # Read and extract text
    content = await file.read()

    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "message": f"File exceeds {settings.max_file_size_mb}MB limit.",
            },
        )

    extracted = extract_text(content, file.filename)

    # Classify
    classification = classify_document(extracted.text)

    return {
        "document_type": classification.document_type.value,
        "confidence": classification.confidence,
        "reasoning": classification.reasoning,
        "key_signals": classification.key_signals,
    }


# --- Static files (upload UI) ---

_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/ui", StaticFiles(directory=_static_dir, html=True), name="static")


# --- Root ---


@app.get("/")
async def root():
    """Root endpoint with links to docs and UI."""
    return {
        "message": "Auto-Triage & Document Extractor Agent",
        "docs": "/docs",
        "ui": "/ui",
        "health": "/api/v1/health",
    }
