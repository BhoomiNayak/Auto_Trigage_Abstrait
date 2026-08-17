"""LLM service for structured document extraction using Google Gemini."""

import json
from typing import Union

from google import generativeai as genai

from app.config import get_settings
from app.models.base import DocumentType
from app.models.invoice import InvoiceExtraction
from app.models.ticket import TicketExtraction
from app.models.resume import ResumeExtraction


class LLMError(Exception):
    """Raised when LLM extraction fails."""

    pass


# --- Prompt Templates ---

INVOICE_PROMPT = """You are an expert document extraction agent. Extract structured data from the following invoice text.

RULES:
- Extract all fields you can find. Use null for fields not present in the document.
- For dates, use ISO format (YYYY-MM-DD).
- For currency amounts, use numeric values without currency symbols.
- Determine if the invoice is overdue based on the due date (assume today's context).
- Assign a priority: critical (overdue + large amount >$5000), high (due within 3 days or amount >$5000), medium (due within 30 days), low (no urgency).
- Assign a category from: accounts_payable, consulting_services, software_licenses, office_supplies, utilities, maintenance, other.

CONFIDENCE SCORING RULES:
- Rate overall confidence (0.0-1.0) based on how clearly the information is stated in the document.
- 0.9-1.0: Field value is explicitly and unambiguously stated.
- 0.7-0.9: Field value is clearly present but requires minor interpretation.
- 0.5-0.7: Field value is inferred or partially visible.
- Below 0.5: Field value is a best guess.
- List ALL field names where your confidence is below 0.7 in low_confidence_fields.
- Be honest — do not default to 1.0. Most real extractions have some uncertainty.

- Provide a one-line reasoning for the priority and category assignment.

DOCUMENT TEXT:
---
{text}
---

Return ONLY a valid JSON object matching this exact schema:
{{
  "document_type": "invoice",
  "triage": {{
    "priority": "critical|high|medium|low",
    "category": "string",
    "reasoning": "string"
  }},
  "confidence": {{
    "overall_confidence": 0.0-1.0,
    "low_confidence_fields": ["field1", "field2"]
  }},
  "raw_text_preview": "first 500 chars of document",
  "data": {{
    "vendor_name": "string",
    "vendor_address": "string or null",
    "vendor_contact": "string or null",
    "invoice_number": "string",
    "invoice_date": "YYYY-MM-DD or null",
    "due_date": "YYYY-MM-DD or null",
    "bill_to": "string or null",
    "bill_to_address": "string or null",
    "line_items": [{{"description": "string", "quantity": number_or_null, "unit_price": number_or_null, "total": number}}],
    "subtotal": number_or_null,
    "tax_amount": number_or_null,
    "total_amount": number,
    "currency": "USD",
    "payment_terms": "string or null",
    "payment_method": "string or null",
    "is_overdue": true/false
  }}
}}"""

TICKET_PROMPT = """You are an expert support ticket triage agent. Extract structured data from the following support ticket.

RULES:
- Extract all fields you can find. Use null for fields not present.
- Determine customer sentiment: angry, frustrated, neutral, or satisfied.
- Check if this is an escalation (customer threatening to leave, requesting manager, or expressing extreme dissatisfaction).
- Assign issue_category: billing, technical, account, product_feedback, shipping, general_inquiry, cancellation, or other.
- Suggest a response type: apologize_and_refund, escalate_to_engineering, escalate_to_manager, provide_instructions, request_more_info, acknowledge_feedback, process_cancellation, or other.
- Assign priority: critical (angry + escalation + revenue risk), high (frustrated + billing/cancellation), medium (neutral + standard issue), low (feedback/general inquiry).
- Assign a category from: billing_escalation, technical_support, account_management, product_feedback, shipping_issue, general_inquiry, churn_risk, other.

CONFIDENCE SCORING RULES:
- Rate overall confidence (0.0-1.0) based on how clearly the information is stated in the document.
- List ALL field names where your confidence is below 0.7 in low_confidence_fields.

- Provide a one-line reasoning for the priority and category assignment.

DOCUMENT TEXT:
---
{text}
---

Return ONLY a valid JSON object matching this exact schema:
{{
  "document_type": "support_ticket",
  "triage": {{
    "priority": "critical|high|medium|low",
    "category": "string",
    "reasoning": "string"
  }},
  "confidence": {{
    "overall_confidence": 0.0-1.0,
    "low_confidence_fields": ["field1", "field2"]
  }},
  "raw_text_preview": "first 500 chars of document",
  "data": {{
    "customer_name": "string or null",
    "customer_email": "string or null",
    "customer_id": "string or null",
    "subject": "string",
    "summary": "string (2-3 sentences max)",
    "issue_category": "billing|technical|account|product_feedback|shipping|general_inquiry|cancellation|other",
    "sentiment": "angry|frustrated|neutral|satisfied",
    "is_escalation": true/false,
    "product_mentioned": "string or null",
    "order_number": "string or null",
    "requested_action": "string or null",
    "suggested_response_type": "string"
  }}
}}"""

RESUME_PROMPT = """You are an expert resume/CV analysis agent. Extract structured data from the following resume.

RULES:
- Extract all fields you can find. Use null for fields not present.
- For experiences, include max 5 entries (most recent first), with max 3 highlights each.
- Estimate total years of experience from the work history.
- Separate technical skills (programming, tools, frameworks) from soft skills (leadership, communication).
- Provide a fit_score (0.0-1.0) based on overall candidate strength: depth of experience, skill diversity, career progression, and education quality.
- Provide fit_reasoning explaining the score.
- Assign priority: high (fit_score > 0.8), medium (0.5-0.8), low (< 0.5). Never use critical for resumes.
- Assign a category based on apparent role type: engineering_hire, design_hire, product_hire, marketing_hire, operations_hire, executive_hire, entry_level, other.

CONFIDENCE SCORING RULES:
- Rate overall confidence (0.0-1.0) based on how clearly the information is stated in the document.
- List ALL field names where your confidence is below 0.7 in low_confidence_fields.

- Provide a one-line reasoning for the priority and category assignment.

DOCUMENT TEXT:
---
{text}
---

Return ONLY a valid JSON object matching this exact schema:
{{
  "document_type": "resume",
  "triage": {{
    "priority": "high|medium|low",
    "category": "string",
    "reasoning": "string"
  }},
  "confidence": {{
    "overall_confidence": 0.0-1.0,
    "low_confidence_fields": ["field1", "field2"]
  }},
  "raw_text_preview": "first 500 chars of document",
  "data": {{
    "full_name": "string",
    "email": "string or null",
    "phone": "string or null",
    "location": "string or null",
    "linkedin_url": "string or null",
    "portfolio_url": "string or null",
    "summary": "string or null (max 300 chars)",
    "technical_skills": ["skill1", "skill2"],
    "soft_skills": ["skill1", "skill2"],
    "total_years_experience": number_or_null,
    "experiences": [
      {{
        "company": "string",
        "title": "string",
        "duration": "string or null",
        "highlights": ["string"]
      }}
    ],
    "education": [
      {{
        "institution": "string",
        "degree": "string",
        "year": "string or null"
      }}
    ],
    "certifications": ["string"],
    "languages": ["string"],
    "fit_score": 0.0-1.0,
    "fit_reasoning": "string"
  }}
}}"""


# --- Service Functions ---


def _get_model():
    """Create a Gemini generative model client."""
    settings = get_settings()
    if not settings.is_configured:
        raise LLMError(
            "GEMINI_API_KEY is not configured. "
            "Set it in .env file or as environment variable."
        )
    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel(
        model_name=settings.gemini_model,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )


def _sanitize_text(text: str, max_chars: int = 15000) -> str:
    """Truncate and sanitize document text before sending to LLM."""
    return text[:max_chars]


def _parse_response(response_text: str, model_class, raw_text: str = ""):
    """Parse JSON response from Gemini and validate against Pydantic model."""
    try:
        data = json.loads(response_text)
        # Ensure raw_text_preview doesn't exceed max_length
        if "raw_text_preview" in data:
            data["raw_text_preview"] = data["raw_text_preview"][:500]
        return model_class.model_validate(data)
    except json.JSONDecodeError as e:
        raise LLMError(f"LLM returned invalid JSON: {e}")
    except Exception as e:
        raise LLMError(f"Response validation failed: {e}")


def extract_invoice(text: str) -> InvoiceExtraction:
    """Extract structured data from an invoice document."""
    model = _get_model()
    settings = get_settings()

    prompt = INVOICE_PROMPT.format(text=_sanitize_text(text))

    for attempt in range(settings.max_retries + 1):
        try:
            response = model.generate_content(prompt)
            result = _parse_response(response.text, InvoiceExtraction)
            result.raw_text_preview = text[:500]
            return result
        except LLMError:
            if attempt == settings.max_retries:
                raise
        except Exception as e:
            if attempt == settings.max_retries:
                raise LLMError(f"Invoice extraction failed: {e}")


def extract_ticket(text: str) -> TicketExtraction:
    """Extract structured data from a support ticket."""
    model = _get_model()
    settings = get_settings()

    prompt = TICKET_PROMPT.format(text=_sanitize_text(text))

    for attempt in range(settings.max_retries + 1):
        try:
            response = model.generate_content(prompt)
            result = _parse_response(response.text, TicketExtraction)
            result.raw_text_preview = text[:500]
            return result
        except LLMError:
            if attempt == settings.max_retries:
                raise
        except Exception as e:
            if attempt == settings.max_retries:
                raise LLMError(f"Ticket extraction failed: {e}")


def extract_resume(text: str) -> ResumeExtraction:
    """Extract structured data from a resume/CV."""
    model = _get_model()
    settings = get_settings()

    prompt = RESUME_PROMPT.format(text=_sanitize_text(text))

    for attempt in range(settings.max_retries + 1):
        try:
            response = model.generate_content(prompt)
            result = _parse_response(response.text, ResumeExtraction)
            result.raw_text_preview = text[:500]
            return result
        except LLMError:
            if attempt == settings.max_retries:
                raise
        except Exception as e:
            if attempt == settings.max_retries:
                raise LLMError(f"Resume extraction failed: {e}")


def extract_document(
    text: str, doc_type: DocumentType
) -> Union[InvoiceExtraction, TicketExtraction, ResumeExtraction]:
    """Extract structured data based on document type.

    This is the main entry point for the LLM service.
    """
    extractors = {
        DocumentType.invoice: extract_invoice,
        DocumentType.support_ticket: extract_ticket,
        DocumentType.resume: extract_resume,
    }

    extractor = extractors.get(doc_type)
    if extractor is None:
        raise ValueError(
            f"Unsupported document type for extraction: '{doc_type}'. "
            f"Supported: {', '.join(t.value for t in extractors.keys())}"
        )

    return extractor(text)
