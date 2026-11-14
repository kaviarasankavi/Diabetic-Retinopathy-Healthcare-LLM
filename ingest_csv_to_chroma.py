"""
Standalone ingestion script that bypasses sentence_transformers.

Uses transformers + chromadb directly to avoid the ONNX mutex.cc deadlock
in sentence_transformers on this macOS system.

This script:
  1. Loads all documents (PDFs, CSV patients)
  2. Splits into chunks
  3. Embeds using transformers AutoModel (pure PyTorch, no ONNX)
  4. Stores in ChromaDB with the same collection name as app.py expects

Usage:
    TOKENIZERS_PARALLELISM=false python3 ingest_csv_to_chroma.py
"""

import os
import csv
import glob
import shutil
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import chromadb

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ─── Configuration ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
PATIENT_CSV = os.path.join(DOCUMENTS_DIR, "diabetic_retinopathy_patients.csv")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 128




# ─── Embedding Helper ──────────────────────────────────────────────

def mean_pooling(model_output, attention_mask):
    """Mean pooling - take attention mask into account for averaging."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class DirectEmbedder:
    """Embedder using transformers directly — no sentence_transformers needed."""

    def __init__(self, model_name):
        print(f"  Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"  Loading model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        print(f"  ✓ Model ready")

    def embed_batch(self, texts, batch_size=BATCH_SIZE):
        """Embed a list of texts in batches, returning normalized numpy arrays."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=256, return_tensors='pt'
            )
            with torch.no_grad():
                outputs = self.model(**encoded)
            embeddings = mean_pooling(outputs, encoded['attention_mask'])
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.numpy())
        return np.concatenate(all_embeddings, axis=0)


# ─── Document Loaders ──────────────────────────────────────────────

def load_pdf_documents():
    """Load all PDF files from the documents directory."""
    documents = []
    pdf_files = glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf"))
    if not pdf_files:
        print("  ⚠ No PDF files found")
        return documents

    print(f"\n📄 Loading {len(pdf_files)} PDF(s)...")
    for filepath in sorted(pdf_files):
        filename = os.path.basename(filepath)
        try:
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["type"] = "research_paper"
                doc.metadata["filename"] = filename
            documents.extend(docs)
            print(f"  ✓ {filename} ({len(docs)} pages)")
        except Exception as e:
            print(f"  ✗ Failed: {filename}: {e}")
    return documents


def load_patient_records():
    """Load patient records from CSV."""
    documents = []
    if not os.path.isfile(PATIENT_CSV):
        print("  ⚠ No diabetic_retinopathy_patients.csv found")
        return documents

    print(f"\n🏥 Loading patient records from CSV...")
    with open(PATIENT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["PatientID"]
            content = (
                f"Patient Record: {pid}\n"
                f"Age: {row['Age']} years | Gender: {row['Gender']}\n"
                f"Diabetes Type: {row['DiabetesType']} | Duration: {row['DiabetesDuration_Years']} years\n\n"
                f"Lab Results:\n"
                f"  HbA1c: {row['HbA1c_Percent']}% | Fasting Blood Sugar: {row['FastingBloodSugar_mg_dL']} mg/dL\n"
                f"  BMI: {row['BMI']} | Blood Pressure: {row['BloodPressure_mmHg']} mmHg\n\n"
                f"Diabetic Retinopathy Assessment:\n"
                f"  DR Stage: {row['DR_Stage']}\n"
                f"  Affected Eye: {row['AffectedEye']}\n"
                f"  Visual Acuity: {row['VisualAcuity_Snellen']}\n"
                f"  Intraocular Pressure: {row['IntraocularPressure_mmHg']} mmHg\n"
                f"  Macular Edema: {row['MacularEdema']}\n\n"
                f"Risk Factors:\n"
                f"  Smoking Status: {row['SmokingStatus']}\n"
                f"  Family History of Diabetes: {row['FamilyHistoryDiabetes']}\n"
                f"  Comorbidities: {row['Comorbidities']}\n\n"
                f"Treatment & Management:\n"
                f"  Current Medications: {row['CurrentMedications']}\n"
                f"  Treatment Given: {row['TreatmentGiven']}\n"
                f"  Follow-up Interval: {row['FollowUpInterval']}\n"
            )
            doc = Document(
                page_content=content,
                metadata={
                    "type": "patient",
                    "filename": "diabetic_retinopathy_patients.csv",
                    "source": PATIENT_CSV,
                    "patient_id": pid,
                    "dr_stage": row["DR_Stage"],
                    "diabetes_type": row["DiabetesType"],
                    "age": int(row["Age"]),
                },
            )
            documents.append(doc)

    # Stage breakdown
    stage_counts = {}
    for doc in documents:
        stage = doc.metadata.get("dr_stage", "Unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    print(f"  ✓ {len(documents)} patient records loaded")
    for stage, count in sorted(stage_counts.items()):
        print(f"     • {stage}: {count}")
    return documents





# ─── Main ──────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("   DR Self-RAG — Direct Ingestion (no ONNX)")
    print("=" * 55)

    # Step 1: Load documents
    all_docs = []
    all_docs.extend(load_pdf_documents())
    all_docs.extend(load_patient_records())


    if not all_docs:
        print("\n❌ No documents found!")
        return

    # Count by type
    type_counts = {}
    for doc in all_docs:
        t = doc.metadata.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\n📊 Documents by type:")
    for t, c in sorted(type_counts.items()):
        print(f"   • {t}: {c}")

    # Step 2: Split
    print(f"\n✂️  Splitting (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        length_function=len, separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"  ✓ {len(chunks)} chunks from {len(all_docs)} documents")

    type_chunk_counts = {}
    for c in chunks:
        t = c.metadata.get("type", "unknown")
        type_chunk_counts[t] = type_chunk_counts.get(t, 0) + 1
    for t, c in sorted(type_chunk_counts.items()):
        print(f"   • {t}: {c} chunks")

    # Step 3: Load embedding model
    print(f"\n🧠 Loading embedding model (PyTorch, no ONNX)...")
    t0 = time.time()
    embedder = DirectEmbedder(EMBEDDING_MODEL)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Step 4: Embed all chunks
    texts = [c.page_content for c in chunks]
    print(f"\n📐 Embedding {len(texts)} chunks in batches of {BATCH_SIZE}...")

    t0 = time.time()
    embeddings = embedder.embed_batch(texts)
    elapsed = time.time() - t0
    print(f"  ✓ Embedded in {elapsed:.1f}s ({len(texts) / elapsed:.0f} chunks/sec)")

    # Step 5: Store in ChromaDB
    print(f"\n💾 Writing to ChromaDB at '{CHROMA_DB_DIR}'...")
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
        print("  ↻ Cleared existing vector store")

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.create_collection(
        name="diabetic_retinopathy",
        metadata={"hnsw:space": "cosine"},
    )

    # Insert in batches
    CHROMA_BATCH = 500
    for i in range(0, len(chunks), CHROMA_BATCH):
        end = min(i + CHROMA_BATCH, len(chunks))
        batch_chunks = chunks[i:end]
        batch_embeddings = embeddings[i:end].tolist()

        ids = [f"chunk_{j}" for j in range(i, end)]
        documents_text = [c.page_content for c in batch_chunks]
        metadatas = []
        for c in batch_chunks:
            # ChromaDB metadata must be str/int/float/bool
            meta = {}
            for k, v in c.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            metadatas.append(meta)

        collection.add(
            ids=ids,
            embeddings=batch_embeddings,
            documents=documents_text,
            metadatas=metadatas,
        )
        print(f"  ✓ Batch {i // CHROMA_BATCH + 1}/{(len(chunks) + CHROMA_BATCH - 1) // CHROMA_BATCH} — {end}/{len(chunks)} chunks stored")

    print(f"\n✅ Ingestion complete!")
    print(f"  • Documents: {len(all_docs)}")
    print(f"  • Chunks: {len(chunks)}")
    print(f"  • Types: {', '.join(sorted(type_chunk_counts.keys()))}")
    print(f"  • Vector store: {CHROMA_DB_DIR}")
    print(f"  • Collection: diabetic_retinopathy ({collection.count()} vectors)")
    print(f"\nRun 'python app.py' to start asking questions.\n")


if __name__ == "__main__":
    main()
