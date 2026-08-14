"""Pydantic models for support ticket document extraction."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.models.base import DocumentType, ExtractionResult


class Sentiment(str, Enum):
    """Customer emotional tone."""

    angry = "angry"
    frustrated = "frustrated"
    neutral = "neutral"
    satisfied = "satisfied"


class IssueCategory(str, Enum):
    """Support ticket issue categories."""

    billing = "billing"
    technical = "technical"
    account = "account"
    product_feedback = "product_feedback"
    shipping = "shipping"
    general_inquiry = "general_inquiry"
    cancellation = "cancellation"
    other = "other"


class TicketData(BaseModel):
    """Structured data extracted from a support ticket."""

    # Customer info
    customer_name: Optional[str] = Field(None, description="Name of the customer")
    customer_email: Optional[str] = Field(None, description="Customer's email address")
    customer_id: Optional[str] = Field(None, description="Customer account ID if mentioned")

    # Ticket content
    subject: str = Field(..., description="Brief subject/title of the issue")
    summary: str = Field(
        ...,
        max_length=300,
        description="2-3 sentence summary of the customer's issue",
    )
    issue_category: IssueCategory = Field(..., description="Category of the issue")

    # Sentiment & urgency
    sentiment: Sentiment = Field(..., description="Customer's emotional tone")
    is_escalation: bool = Field(
        default=False,
        description="Whether the customer is requesting escalation or threatening to leave",
    )

    # Actionable details
    product_mentioned: Optional[str] = Field(
        None, description="Specific product/service referenced"
    )
    order_number: Optional[str] = Field(
        None, description="Order or reference number if mentioned"
    )
    requested_action: Optional[str] = Field(
        None,
        description="What the customer is asking for (refund, fix, info, etc.)",
    )

    # Suggested response
    suggested_response_type: str = Field(
        ...,
        description="Recommended response approach (e.g., 'apologize_and_refund', 'escalate_to_engineering', 'provide_instructions')",
    )


class TicketExtraction(ExtractionResult):
    """Complete support ticket extraction response."""

    document_type: DocumentType = DocumentType.support_ticket
    data: TicketData
