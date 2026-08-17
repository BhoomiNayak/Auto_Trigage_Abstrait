"""Document classification service for auto-detecting document types using Google Gemini."""

import json

from google import generativeai as genai
from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.base import DocumentType


class ClassificationResult(BaseModel):
    """Result of document type classification."""

    document_type: DocumentType = Field(
        ...,
        description="Detected document type: invoice, support_ticket, resume, or unknown",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification confidence (0.0 = guessing, 1.0 = certain)",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this type was chosen based on key signals found",
    )
    key_signals: list[str] = Field(
        default_factory=list,
        description="Specific phrases or patterns that led to this classification",
    )


CLASSIFICATION_PROMPT = """You are an expert document classification agent. Your job is to determine the type of document based on its content.

DOCUMENT TYPES:
1. **invoice** — Bills, invoices, purchase orders. Key signals: amounts, line items, vendor info, "invoice number", "bill to", "due date", "total", "payment terms", "subtotal", "tax".
2. **support_ticket** — Customer complaints, help requests, bug reports. Key signals: "issue", "problem", "help", customer name/email, complaint language, urgency words, "order number", emotional tone.
3. **resume** — CVs, resumes, professional profiles. Key signals: person's name at top, "experience", "education", "skills", job titles, company names, dates of employment, "certifications".
4. **unknown** — Only use this if the document genuinely doesn't fit any of the above categories.

RULES:
- Analyze the text structure, keywords, and overall purpose.
- Identify at least 2-3 key signals (specific words/phrases) that support your classification.
- Be confident — most business documents clearly fall into one category.
- Only return "unknown" if you truly cannot determine the type (confidence < 0.3).

DOCUMENT TEXT (first 800 characters):
---
{text}
---

Return ONLY a valid JSON object matching this exact schema:
{{
  "document_type": "invoice|support_ticket|resume|unknown",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation",
  "key_signals": ["signal1", "signal2", "signal3"]
}}"""


def classify_document(text: str) -> ClassificationResult:
    """Classify a document's type based on its text content.

    Uses the first 800 characters for efficiency.

    Args:
        text: Extracted document text.

    Returns:
        ClassificationResult with type, confidence, reasoning, and key signals.

    Raises:
        RuntimeError: If classification fails.
    """
    settings = get_settings()
    if not settings.is_configured:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. "
            "Set it in .env file or as environment variable."
        )

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    # Use first 800 chars — enough for reliable classification
    preview = text[:800]
    prompt = CLASSIFICATION_PROMPT.format(text=preview)

    try:
        response = model.generate_content(prompt)
        data = json.loads(response.text)
        return ClassificationResult.model_validate(data)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Document classification failed: Invalid JSON response: {e}")
    except Exception as e:
        raise RuntimeError(f"Document classification failed: {e}")
