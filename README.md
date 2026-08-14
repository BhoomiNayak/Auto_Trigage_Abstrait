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
│  │   Service       │  │   Service        │  │  (Instructor) │  │
│  │                 │  │                  │  │               │  │
│  │  PDF → Text     │  │  Text → Type     │  │  Text → JSON  │  │
│  │  TXT → Text     │  │  (8b model)      │  │  (70b model)  │  │
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
│   │  PyMuPDF    │    │         Groq API (Free Tier)          │   │
│   │  (fitz)     │    │                                      │   │
│   │             │    │  llama-3.1-8b-instant (fast/classify) │   │
│   │  PDF parse  │    │  llama-3.3-70b-versatile (accurate)   │   │
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
│  Classify type   │  LLM detects: invoice | ticket | resume
│  (if no hint)    │  Uses fast model (8b), first 800 chars
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Extract data    │  Instructor sends schema-constrained prompt
│                  │  to Groq, receives validated Pydantic object
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
| LLM Provider | **Groq** (free tier) | Fast inference, free API, Llama 3 models |
| Structured Output | **Instructor + Pydantic v2** | Type-safe LLM responses with validation and retry |
| PDF Extraction | **PyMuPDF (fitz)** | Fast, pure Python, no Java dependency |
| Configuration | **pydantic-settings** | Type-safe env loading with validation |
| Deployment | **Docker** | Single container, any cloud |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Groq API key (free at [console.groq.com/keys](https://console.groq.com/keys))

### 1. Clone and install

```bash
git clone https://github.com/your-username/abstrait_auto_triage.git
cd abstrait_auto_triage
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your Groq API key
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
    "priority": "high",
    "category": "consulting_services",
    "reasoning": "Invoice for $10,090.50 due within 14 days from a consulting vendor"
  },
  "confidence": {
    "overall_confidence": 0.92,
    "low_confidence_fields": ["vendor_contact"]
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
    "is_overdue": false
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
    "overall_confidence": 0.95,
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
│   │   ├── classifier.py         # Document type detection
│   │   └── llm.py                # Groq + Instructor extraction
│   └── static/
│       └── index.html            # Upload UI (drag-and-drop)
├── samples/                       # Sample documents for testing
│   ├── invoice_sample.txt
│   ├── ticket_sample.txt
│   └── resume_sample.txt
├── docs/                          # Planning documentation
│   ├── PROJECT_OVERVIEW.md
│   ├── ARCHITECTURE.md
│   ├── DATA_MODELS.md
│   ├── API_DESIGN.md
│   └── IMPLEMENTATION_PLAN.md
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .env.example
├── requirements.txt
├── Procfile                       # For Railway/Render deployment
├── runtime.txt
└── README.md
```

---

## Deployment

### Docker

```bash
docker build -t auto-triage .
docker run -d -p 8000:8000 -e GROQ_API_KEY=gsk_your_key auto-triage
```

### Railway (recommended for demo)

1. Push to GitHub
2. [railway.app](https://railway.app) → New Project → Deploy from repo
3. Add env var: `GROQ_API_KEY`
4. Auto-deploys from Dockerfile

### Render

1. Push to GitHub
2. [render.com](https://render.com) → New Web Service
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add env var: `GROQ_API_KEY`

---

## Security

| Concern | Mitigation |
|---------|-----------|
| Prompt injection | System prompts instruct to ignore embedded instructions; input truncated to 15k chars |
| File upload attacks | Extension validation + PDF magic byte check |
| API key exposure | `.env` gitignored; error messages sanitized (no key leakage) |
| Large file DoS | 10MB file size limit enforced before processing |
| Error information leakage | LLM errors return safe generic messages, not raw exceptions |

---

## Design Decisions

| Decision | Reasoning |
|----------|-----------|
| Groq over OpenAI | Free tier, fast inference, good enough for demo |
| Instructor over raw JSON | Automatic validation, retries on schema mismatch, type safety |
| Separate classifier from extractor | Allows skip-classification when type is known; lighter prompt for routing |
| Two-model strategy | 8b for fast tasks (classify), 70b for precision (extract) |
| Pydantic v2 everywhere | Native FastAPI integration, fast serialization, clear schemas |
| No database in MVP | Stateless API is simpler; persistence is one config change away |

---

## How This Maps to Abstrabit's Value Proposition

| Abstrabit Pain Point | How This Agent Solves It |
|---------------------|--------------------------|
| **Spreadsheet Hell** | Structured JSON replaces manual data entry entirely |
| **Manual Ops Bottleneck** | Full automation — zero human intervention for standard docs |
| **Fragile Infrastructure** | Clean API with validation; no brittle scripts |
| **Tribal Knowledge** | Rules encoded in prompts and schemas, not in someone's head |

---

## Future Enhancements

- OCR support (pytesseract) for scanned PDFs
- Webhook callbacks after extraction
- Custom schemas via API (user-defined fields)
- Multi-language support
- Analytics dashboard (extraction volumes, accuracy tracking)
- PostgreSQL persistence layer
- API key authentication for production

---

## License

Built for demonstration purposes as part of an application to [Abstrabit](https://www.abstrabit.com/).
