"""Shared base models, enums, and common fields used across all document types."""

from enum import Enum

from pydantic import BaseModel, Field


class Priority(str, Enum):
    """Document urgency level."""

    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"


class DocumentType(str, Enum):
    """Supported document types."""

    invoice = "invoice"
    support_ticket = "support_ticket"
    resume = "resume"
    unknown = "unknown"


class FieldConfidence(BaseModel):
    """Confidence metadata for extracted fields."""

    overall_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the extraction (0.0 = no confidence, 1.0 = certain)",
    )
    low_confidence_fields: list[str] = Field(
        default_factory=list,
        description="List of field names where extraction confidence is below 0.7",
    )


class TriageMetadata(BaseModel):
    """Auto-triage tags applied to every document."""

    priority: Priority = Field(
        ...,
        description="Urgency level: critical, high, medium, or low",
    )
    category: str = Field(
        ...,
        description="Business category (e.g., 'accounts_payable', 'technical_support', 'engineering_hire')",
    )
    reasoning: str = Field(
        ...,
        description="One-line explanation of why this priority and category were assigned",
    )


class ExtractionResult(BaseModel):
    """Base class for all extraction responses."""

    document_type: DocumentType
    triage: TriageMetadata
    confidence: FieldConfidence
    raw_text_preview: str = Field(
        ...,
        max_length=500,
        description="First 500 characters of extracted text for reference",
    )
