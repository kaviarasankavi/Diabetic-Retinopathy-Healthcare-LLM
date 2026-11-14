"""
Document Ingestion Script for Diabetic Retinopathy Self-RAG Application.

This script reads all documents from the 'documents/' folder:
  - PDFs (research papers, clinical guidelines) via PyPDFLoader
  - Patient records (CSV) from diabetic_retinopathy_patients.csv

All content is split into chunks, embedded using HuggingFace
sentence-transformers, and stored in a ChromaDB vector database
with metadata tags for document type, filename, and source.

Document types:
  - research_paper    : PDF research papers and clinical guidelines
  - patient           : Patient medical records (CSV — 501 patients)
"""

import os
import csv
import glob
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# Configuration
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
PATIENT_CSV = os.path.join(DOCUMENTS_DIR, "diabetic_retinopathy_patients.csv")




def load_pdf_documents(documents_dir: str) -> list:
    """Load all PDF files from the documents directory."""
    documents = []
    pdf_files = glob.glob(os.path.join(documents_dir, "*.pdf"))

    if not pdf_files:
        console.print("[yellow]  ⚠ No PDF files found[/yellow]")
        return documents

    console.print(f"\n[cyan]📄 Loading {len(pdf_files)} PDF(s)...[/cyan]")

    for filepath in sorted(pdf_files):
        filename = os.path.basename(filepath)
        try:
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["type"] = "research_paper"
                doc.metadata["filename"] = filename
            documents.extend(docs)
            console.print(
                f"  [green]✓[/green] [{'research_paper':>15}] {filename} "
                f"[dim]({len(docs)} pages)[/dim]"
            )
        except Exception as e:
            console.print(f"  [red]✗[/red] Failed to load {filename}: {e}")

    return documents


def load_patient_records(documents_dir: str) -> list:
    """Load patient records from diabetic_retinopathy_patients.csv.

    Each CSV row is converted into a rich text document suitable for
    semantic search and RAG retrieval. Patient IDs are used (no names).
    """
    documents = []
    csv_path = PATIENT_CSV

    if not os.path.isfile(csv_path):
        console.print("[yellow]  ⚠ No diabetic_retinopathy_patients.csv found[/yellow]")
        return documents

    console.print(f"\n[cyan]🏥 Loading patient records from CSV...[/cyan]")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row_count = 0

        for row in reader:
            row_count += 1
            pid = row["PatientID"]

            # Build a rich, searchable text document from each patient row
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
                    "source": csv_path,
                    "patient_id": pid,
                    "dr_stage": row["DR_Stage"],
                    "diabetes_type": row["DiabetesType"],
                    "age": int(row["Age"]),
                },
            )
            documents.append(doc)

    console.print(
        f"  [green]✓[/green] [{'patient':>15}] "
        f"{row_count} patient records loaded from CSV"
    )

    # Show DR stage breakdown
    stage_counts = {}
    for doc in documents:
        stage = doc.metadata.get("dr_stage", "Unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    for stage, count in sorted(stage_counts.items()):
        console.print(f"       [dim]• {stage}: {count} patients[/dim]")

    return documents





def split_documents(documents: list) -> list:
    """Split documents into smaller chunks for embedding, preserving metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks: list) -> Chroma:
    """Create ChromaDB vector store from document chunks using batched insertion."""
    console.print(f"\n[cyan] Loading embedding model: '{EMBEDDING_MODEL}'...[/cyan]")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    console.print(f"[cyan] Creating vector store at '{CHROMA_DB_DIR}'...[/cyan]")

    # Remove existing DB if present
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
        console.print("[yellow]  ↻ Cleared existing vector store[/yellow]")

    # Create vector store with the first batch
    BATCH_SIZE = 500
    total = len(chunks)

    console.print(f"[cyan]  Embedding {total} chunks in batches of {BATCH_SIZE}...[/cyan]")

    # First batch creates the store
    first_batch = chunks[:BATCH_SIZE]
    vector_store = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name="diabetic_retinopathy",
    )
    console.print(f"  [green]✓[/green] Batch 1/{(total + BATCH_SIZE - 1) // BATCH_SIZE} — {min(BATCH_SIZE, total)}/{total} chunks embedded")

    # Remaining batches are added to the existing store
    for i in range(BATCH_SIZE, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vector_store.add_documents(batch)
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        console.print(f"  [green]✓[/green] Batch {batch_num}/{total_batches} — {min(i + BATCH_SIZE, total)}/{total} chunks embedded")

    return vector_store



def main():
    console.print("\n[bold magenta]═══════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]   DR Self-RAG — Unified Document Ingestion            [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════════════[/bold magenta]")

    # Step 1: Load all document types
    all_documents = []

    # Load PDFs (research papers & guidelines)
    pdf_docs = load_pdf_documents(DOCUMENTS_DIR)
    all_documents.extend(pdf_docs)

    # Load patient records
    patient_docs = load_patient_records(DOCUMENTS_DIR)
    all_documents.extend(patient_docs)


    if not all_documents:
        console.print("[red]No documents to process. Exiting.[/red]")
        return

    # Summary
    console.print(f"\n[cyan] Documents loaded by type:[/cyan]")
    type_counts = {}
    for doc in all_documents:
        doc_type = doc.metadata.get("type", "unknown")
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    for dtype, count in sorted(type_counts.items()):
        console.print(f"   [dim]•[/dim] {dtype}: {count} document(s)")

    # Step 2: Split into chunks
    console.print(
        f"\n[cyan]✂️  Splitting documents into chunks "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...[/cyan]"
    )
    chunks = split_documents(all_documents)
    console.print(
        f"  [green]✓[/green] Created {len(chunks)} chunks "
        f"from {len(all_documents)} document(s)"
    )

    # Show chunk breakdown by type
    type_chunk_counts = {}
    for chunk in chunks:
        doc_type = chunk.metadata.get("type", "unknown")
        type_chunk_counts[doc_type] = type_chunk_counts.get(doc_type, 0) + 1

    console.print(f"\n[cyan] Chunks by document type:[/cyan]")
    for dtype, count in sorted(type_chunk_counts.items()):
        console.print(f"   [dim]•[/dim] {dtype}: {count} chunks")

    # Step 3: Create vector store
    vector_store = create_vector_store(chunks)

    console.print(f"\n[bold green] Ingestion complete![/bold green]")
    console.print(f"  • Documents processed: {len(all_documents)}")
    console.print(f"  • Chunks created: {len(chunks)}")
    console.print(f"  • Document types: {', '.join(sorted(type_chunk_counts.keys()))}")
    console.print(f"  • Vector store saved to: {CHROMA_DB_DIR}")
    console.print(f"  • Embedding model: {EMBEDDING_MODEL}")
    console.print(f"\n[dim]Run 'python app.py' to start asking questions.[/dim]\n")


if __name__ == "__main__":
    main()
