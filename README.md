#  Diabetic Retinopathy Self-RAG Assistant

A **Self-Reflective Retrieval-Augmented Generation (Self-RAG)** application for answering questions about **Diabetic Retinopathy**, with a unified **ChromaDB** knowledge base storing all medical knowledge, patient records, clinical guidelines, reports, and research.

##  Architecture

```
                    ┌──────────────────────────────────────┐
                    │      Unified ChromaDB Vector Store   │
                    │                                      │
                    │   DR Knowledge (5 docs)              │
                    │   Patient Records (300 patients)     │
                    │   Clinical Guidelines (3 docs)       │
                    │   Medical Reports (3 docs)           │
                    │   Research Summaries (3 docs)        │
                    └──────────────┬───────────────────────┘
                                   │
User Question → Embedding → Similarity Search → Top-K Chunks
                                   │
                    ┌──────────────▼───────────────────────┐
                    │      Self-RAG Pipeline (5 Stages)    │
                    │                                      │
                    │  1. Retrieval Decision (need lookup?)│
                    │  2. Relevance Grading (grade chunks) │
                    │  3. Generation (answer from context) │
                    │  4. Hallucination Check (grounded?)  │
                    │  5. Answer Quality (useful?)         │
                    └──────────────┬───────────────────────┘
                                   │
                              Answer + Sources
```

| Component | Technology | Cost |
|-----------|-----------|------|
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Free (local) |
| Vector DB | ChromaDB | Free (local) |
| LLM | Groq (Llama 3.3 70B) | Free tier |
| Framework | LangChain | Free |

##  Project Structure

```
RAG/
├── app.py                          # Self-RAG CLI application
├── ingest.py                       # Unified document ingestion script
├── convert_patients.py             # Patient JSON → text converter
├── documents/                      # All knowledge base documents
│   ├── *.txt                       # DR medical knowledge (5 files)
│   ├── patient_records/            # Patient records (300 files)
│   ├── dr_guidelines/              # Clinical guidelines (3 files)
│   ├── dr_reports/                 # Medical reports (3 files)
│   └── dr_research/                # Research summaries (3 files)
├── chroma_db/                      # Vector database (auto-generated)
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (GROQ_API_KEY)
└── README.md                       # This file
```

##  Knowledge Base

| Category | Files | Description |
|----------|-------|-------------|
| DR Knowledge | 5 | Overview, stages, symptoms/diagnosis, treatment, prevention |
| Patient Records | 300 | Patient demographics, visit histories, prescriptions, diagnoses |
| Clinical Guidelines | 3 | Screening protocols, management pathways, referral criteria |
| Medical Reports | 3 | Sample fundus exam, OCT scan report, case studies |
| Research | 3 | Self-RAG paper summary, MedGemma capabilities, RAG comparison |

##  Setup

### 1. Get a Free Groq API Key

1. Go to [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign up for a free account
3. Create a new API key

### 2. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

```bash
# Edit .env and set your Groq API key
GROQ_API_KEY=your_actual_key_here
```

### 4. Ingest Documents

```bash
python ingest.py
```

### 5. Run the Assistant

```bash
python app.py
```

##  Usage Examples

```
 You: What are the stages of diabetic retinopathy?
   → Retrieves from DR knowledge docs, Self-RAG verifies and responds

 You: What is patient PAT-001's medical history?
   → Retrieves from patient records, shows visits and prescriptions

 You: What are the screening guidelines for DR?
   → Retrieves from clinical guidelines docs

 You: Explain the sample fundus exam findings
   → Retrieves from medical reports

 You: How does Self-RAG compare to traditional RAG?
   → Retrieves from research summaries
```

##  Configuration

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `CHUNK_SIZE` | `ingest.py` | 500 | Size of text chunks |
| `CHUNK_OVERLAP` | `ingest.py` | 50 | Overlap between chunks |
| `EMBEDDING_MODEL` | both | `all-MiniLM-L6-v2` | HuggingFace model |
| `LLM_MODEL` | `app.py` | `llama-3.3-70b-versatile` | Groq LLM model |
| `TOP_K_RESULTS` | `app.py` | 4 | Chunks retrieved per query |
| `EXPANDED_K_RESULTS` | `app.py` | 8 | Fallback expanded retrieval |
| `temperature` | `app.py` | 0.3 | LLM creativity (0-1) |
