# Auto-Triage & Document Extractor Agent

An intelligent document processing agent that ingests invoices, support tickets, and resumes — extracts structured data using LLMs — and auto-assigns priority, category, and confidence scores. Built as a use case demo for [Abstrabit](https://www.abstrabit.com/).

---

## The Problem

Businesses at scale deal with hundreds of documents daily. Today this is manual:

- Someone reads the document
- Copies data into a spreadsheet
- Decides priority based on gut feel
- Routes it to the right team

This is **slow**, **error-prone**, and **doesn't scale**.

---

## The Solution

A FastAPI-based AI agent that:

1. Accepts a PDF or text file
2. Auto-detects the document type
3. Extracts structured fields using an LLM with schema enforcement
4. Tags priority and category with reasoning
5. Returns clean, validated JSON — ready for database insertion or workflow trigger

**Processing time: ~2-4 seconds per document.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│                                                                 │
│    Upload UI (/ui)          API Client           Webhook        │
│    (HTML/JS)                (cURL/Postman)       (Zapier/Make)  │
└────────────┬────────────────────┬────────────────────┬──────────┘
             │                    │                    │
             ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API LAYER (FastAPI)                         │
│                                                                 │
│   POST /api/v1/extract        - Single document extraction      │
│   POST /api/v1/extract/batch  - Batch (up to 10 files)          │
│   POST /api/v1/classify       - Type detection only             │
│   GET  /api/v1/health         - Service health check            │
│                                                                 │
│   Middleware: Request ID, Processing Time, CORS, Error Handling │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER                               │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │   Extractor     │  │   Classifier     │  │  LLM Service  │  │
│  │   Service       │  │   Service        │  │  (Gemini)     │  │
│  │                 │  │                  │  │               │  │
│  │  PDF → Text     │  │  Text → Type     │  │  Text → JSON  │  │
│  │  TXT → Text     │  │  (Gemini Flash)  │  │  (Gemini)     │  │
│  │  MD  → Text     │  │                  │  │               │  │
│  └────────┬────────┘  └────────┬─────────┘  └───────┬───────┘  │
│           │                    │                     │           │
└───────────┼────────────────────┼─────────────────────┼───────────┘
            │                    │                     │
            ▼                    ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                             │
│                                                                 │
│   ┌─────────────┐    ┌──────────────────────────────────────┐   │
│   │  PyMuPDF    │    │      Google Gemini API (Free Tier)    │   │
│   │  (fitz)     │    │                                      │   │
│   │             │    │  gemini-3.6-flash                     │   │
│   │  PDF parse  │    │  (fast, structured JSON output)       │   │
│   └─────────────┘    └──────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PYDANTIC SCHEMA LAYER                         │
│                                                                 │
│   InvoiceExtraction    TicketExtraction    ResumeExtraction      │
│   ├─ vendor_name       ├─ sentiment        ├─ full_name         │
│   ├─ invoice_number    ├─ issue_category   ├─ technical_skills  │
│   ├─ line_items[]      ├─ is_escalation    ├─ experiences[]     │
│   ├─ total_amount      ├─ customer_name    ├─ fit_score         │
│   ├─ due_date          ├─ suggested_resp   ├─ education[]       │
│   └─ is_overdue        └─ order_number     └─ fit_reasoning     │
│                                                                 │
│   + TriageMetadata (priority, category, reasoning)              │
│   + FieldConfidence (overall_confidence, low_confidence_fields) │
└─────────────────────────────────────────────────────────────────┘
```

---

## Request Flow

```
User uploads file
        │
        ▼
┌──────────────────┐
│  Validate file   │  Check extension, size, PDF magic bytes
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Extract text    │  PyMuPDF for PDF, UTF-8 decode for text
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Classify type   │  Gemini detects: invoice | ticket | resume
│  (if no hint)    │  Uses first 800 chars for efficiency
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Extract data    │  Gemini generates structured JSON
│                  │  Validated against Pydantic schema
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Auto-Triage     │  Priority (critical/high/medium/low)
│                  │  Category + reasoning
│                  │  Confidence score per field
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Return JSON     │  Structured, validated, ready for action
└──────────────────┘
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| API Framework | **FastAPI** | Async, auto-generated Swagger docs, production-ready |
| LLM Provider | **Google Gemini** (free tier) | Fast inference, generous rate limits, JSON mode |
| Structured Output | **Gemini JSON mode + Pydantic v2** | Type-safe responses with automatic validation |
| PDF Extraction | **PyMuPDF (fitz)** | Fast, pure Python, no Java dependency |
| Configuration | **pydantic-settings** | Type-safe env loading with validation |
| Deployment | **Render** | Free tier, auto-deploy from GitHub |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Google Gemini API key (free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey))

### 1. Clone and install

```bash
git clone https://github.com/BhoomiNayak/Auto_Trigage_Abstrait.git
cd Auto_Trigage_Abstrait
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

### 3. Run

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Open

- Upload UI: [http://localhost:8000/ui/](http://localhost:8000/ui/)
- API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/extract` | Extract structured data from a single document |
| `POST` | `/api/v1/extract/batch` | Extract from multiple documents (max 10) |
| `POST` | `/api/v1/classify` | Classify document type only (no extraction) |
| `GET` | `/api/v1/health` | Service health check |

---

## Usage Examples

### Extract an invoice (with type hint)

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@samples/invoice_sample.txt" \
  -F "doc_type=invoice"
```

### Auto-detect document type

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@samples/ticket_sample.txt"
```

### Batch extraction

```bash
curl -X POST http://localhost:8000/api/v1/extract/batch \
  -F "files=@samples/invoice_sample.txt" \
  -F "files=@samples/ticket_sample.txt" \
  -F "files=@samples/resume_sample.txt"
```

### Classify only

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -F "file=@samples/resume_sample.txt"
```

---

## Example Output

### Invoice Extraction

```json
{
  "document_type": "invoice",
  "triage": {
    "priority": "critical",
    "category": "consulting_services",
    "reasoning": "Invoice is overdue with a balance greater than $5,000"
  },
  "confidence": {
    "overall_confidence": 0.98,
    "low_confidence_fields": []
  },
  "data": {
    "vendor_name": "Acme Consulting Group",
    "invoice_number": "INV-2024-0847",
    "invoice_date": "2024-11-01",
    "due_date": "2024-11-15",
    "total_amount": 10090.50,
    "currency": "USD",
    "line_items": [
      {
        "description": "Strategy Consulting - October",
        "quantity": 40,
        "unit_price": 150.00,
        "total": 6000.00
      }
    ],
    "payment_terms": "Net 14",
    "is_overdue": true
  }
}
```

### Support Ticket Extraction

```json
{
  "document_type": "support_ticket",
  "triage": {
    "priority": "critical",
    "category": "billing_escalation",
    "reasoning": "Angry customer, double-charged, threatening to cancel, requesting manager"
  },
  "confidence": {
    "overall_confidence": 0.98,
    "low_confidence_fields": []
  },
  "data": {
    "customer_name": "Sarah Mitchell",
    "customer_email": "sarah.mitchell@gmail.com",
    "subject": "Double charged for subscription",
    "sentiment": "angry",
    "is_escalation": true,
    "issue_category": "billing",
    "order_number": "ORD-2024-88432",
    "requested_action": "Immediate refund of duplicate charge",
    "suggested_response_type": "apologize_and_refund"
  }
}
```

---

## Supported Document Types

| Type | Key Extracted Fields | Priority Logic |
|------|---------------------|----------------|
| **Invoice** | Vendor, amount, line items, due date, payment terms | Overdue + >$5k = critical, due soon = high |
| **Support Ticket** | Sentiment, category, escalation, customer info | Angry + escalation = critical, billing = high |
| **Resume** | Skills, experience, education, fit score | fit > 0.8 = high, 0.5-0.8 = medium |

---

## Project Structure

```
abstrait_auto_triage/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Settings (pydantic-settings, .env)
│   ├── main.py                   # FastAPI app, routes, middleware
│   ├── models/
│   │   ├── __init__.py           # Re-exports all models
│   │   ├── base.py               # Priority, DocumentType, TriageMetadata
│   │   ├── invoice.py            # InvoiceData, LineItem
│   │   ├── ticket.py             # TicketData, Sentiment, IssueCategory
│   │   └── resume.py             # ResumeData, Experience, Education
│   ├── services/
│   │   ├── __init__.py
│   │   ├── extractor.py          # PDF/text extraction (PyMuPDF)
│   │   ├── classifier.py         # Document type detection (Gemini)
│   │   └── llm.py                # Google Gemini structured extraction
│   └── static/
│       └── index.html            # Upload UI (drag-and-drop)
├── samples/                       # Sample documents for testing
│   ├── invoice_sample.txt
│   ├── ticket_sample.txt
│   └── resume_sample.txt
├── .env.example                   # Environment variable template
├── .python-version                # Python version for deployment
├── Procfile                       # Render/Heroku start command
├── requirements.txt               # Pinned Python dependencies
└── README.md
```

---

## Key Design Principles

1. **Schema-first** — Every document type has a Pydantic model. The LLM is constrained to output valid JSON matching that schema.

2. **Provider-agnostic LLM layer** — Swapping Gemini for another provider requires changing only the service layer.

3. **Fail gracefully** — Low-confidence extractions are flagged, not silently passed through. Fields the LLM can't extract are marked as `null`.

4. **Auditable** — Every priority/category tag includes a `reasoning` field explaining why.

5. **Stateless API** — Pure function: document in, JSON out. No database required.

---

## Live Demo

**URL:** https://auto-trigage-abstrait.onrender.com/ui

---

## Built By

**Bhoomi Nayak** — SDE Intern Application for [Abstrabit](https://www.abstrabit.com/)
