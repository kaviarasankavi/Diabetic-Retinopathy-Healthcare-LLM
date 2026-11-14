# 🎬 DR Self-RAG Assistant: 3-Minute Demo Script

This demo script showcases the **Self-RAG pipeline** with the unified ChromaDB knowledge base.

## 🛠️ Setup & Recording Preparation

**What to have on your screen:**
- A terminal running `python app.py`
- (Optional) A slide deck or the README diagram for the intro

**Pre-flight Checklist:**
- [ ] Documents are ingested (`python ingest.py`)
- [ ] Self-RAG CLI starts without errors (`python app.py`)

---

## 📝 The Script (Target: 2m 50s)

### Scene 1: Introduction (0:00 - 0:30)
* **Visual:** Start by showing the title of your project.
* **Audio (You say):**
  > "Hello! Today I'm presenting our Diabetic Retinopathy Self-RAG Assistant. This system uses Self-Reflective RAG — a 5-stage pipeline that decides when to retrieve, grades chunk relevance, generates answers, checks for hallucinations, and evaluates quality. All data — medical knowledge, 300 patient records, clinical guidelines, reports, and research — is stored in a single ChromaDB vector database."

### Scene 2: Medical Knowledge Query (0:30 - 1:10)
* **Visual:** Show the terminal with the Self-RAG CLI running.
* **Prompt to type:**
  > "What are the stages of diabetic retinopathy and how do they progress?"
* **Audio (You say):**
  > "Let me ask a medical knowledge question. Watch how the Self-RAG pipeline works — it first decides retrieval is needed, then retrieves 4 chunks from the knowledge base, grades each one for relevance, generates an answer, verifies it's grounded in the sources, and checks if it actually answers the question. All 5 stages are shown transparently."

### Scene 3: Patient Record Query (1:10 - 1:45)
* **Prompt to type:**
  > "What is the medical history of patient PAT-001?"
* **Audio (You say):**
  > "Now I'll query a specific patient record. The system retrieves the patient's information from the same vector database — including their visit history, diagnoses, and prescribed medications across different hospitals. This unified approach means one query can pull together knowledge AND patient data."

### Scene 4: Clinical Guidelines (1:45 - 2:15)
* **Prompt to type:**
  > "When should a diabetic patient be referred urgently to a retina specialist?"
* **Audio (You say):**
  > "Here I'm asking about clinical referral criteria. The system retrieves from the guidelines documents and provides evidence-based referral recommendations. Notice how the source attribution shows which document category each chunk came from — knowledge, patient, guideline, report, or research."

### Scene 5: Closing (2:15 - 2:50)
* **Audio (You say):**
  > "The key innovation is the Self-RAG pipeline's 5 reflection stages. Unlike traditional RAG which retrieves once and generates blindly, Self-RAG grades relevance, checks hallucinations, and verifies answer quality. In our testing, this reduces hallucination rates significantly — achieving 82.5% factual accuracy compared to 78.3% for traditional RAG. All data lives in one ChromaDB vector store, making the system simple to deploy and maintain. Thank you for watching!"

---

## 🔧 Quick Reference

```bash
# Setup
python ingest.py    # Ingest all documents into ChromaDB
python app.py       # Start the Self-RAG CLI

# Sample queries to demo
"What are the stages of diabetic retinopathy?"
"What is the medical history of patient PAT-001?"
"What are the DR screening guidelines?"
"Explain the sample fundus examination findings"
"How does Self-RAG compare to traditional RAG?"
```
