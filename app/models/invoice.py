"""Pydantic models for invoice document extraction."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from app.models.base import DocumentType, ExtractionResult


class LineItem(BaseModel):
    """A single line item on an invoice."""

    description: str = Field(..., description="Item or service description")
    quantity: Optional[float] = Field(None, description="Quantity if specified")
    unit_price: Optional[float] = Field(None, description="Price per unit")
    total: float = Field(..., description="Line item total amount")


class InvoiceData(BaseModel):
    """Structured data extracted from an invoice."""

    # Vendor info
    vendor_name: str = Field(..., description="Name of the vendor/supplier")
    vendor_address: Optional[str] = Field(None, description="Vendor's full address")
    vendor_contact: Optional[str] = Field(None, description="Vendor email or phone")

    # Invoice identifiers
    invoice_number: str = Field(..., description="Unique invoice number/ID")
    invoice_date: Optional[date] = Field(None, description="Date the invoice was issued")
    due_date: Optional[date] = Field(None, description="Payment due date")

    # Recipient
    bill_to: Optional[str] = Field(None, description="Name of the entity being billed")
    bill_to_address: Optional[str] = Field(None, description="Billing address")

    # Financial
    line_items: list[LineItem] = Field(
        default_factory=list,
        description="Itemized charges on the invoice",
    )
    subtotal: Optional[float] = Field(None, description="Sum before tax")
    tax_amount: Optional[float] = Field(None, description="Tax amount")
    total_amount: float = Field(..., description="Total amount due")
    currency: str = Field(default="USD", description="Currency code (e.g., USD, EUR, INR)")

    # Payment
    payment_terms: Optional[str] = Field(
        None, description="Payment terms (e.g., Net 30, Due on receipt)"
    )
    payment_method: Optional[str] = Field(None, description="Accepted payment methods")

    # Status
    is_overdue: bool = Field(
        default=False,
        description="Whether the invoice appears to be past due based on due date",
    )


class InvoiceExtraction(ExtractionResult):
    """Complete invoice extraction response."""

    document_type: DocumentType = DocumentType.invoice
    data: InvoiceData
