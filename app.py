"""
Self-RAG (Self-Reflective Retrieval-Augmented Generation) Application
for Diabetic Retinopathy.

Unified knowledge base stores all data in ChromaDB:
  - Research papers & clinical guidelines (PDFs on DR epidemiology, treatment, screening)
  - Patient records (500 patients from CSV, diabetic retinopathy data)
  - Fundus image metadata (3,662 retinal images across 5 DR severity stages)

Self-RAG adds 5 reflection stages:
  1. Retrieval Decision  — Does this question need knowledge base lookup?
  2. Relevance Grading   — Grade each retrieved chunk for relevance
  3. Generation          — Generate answer from relevant chunks only
  4. Hallucination Check — Is the answer grounded in the context?
  5. Answer Quality      — Does the answer resolve the question?
"""

import os
import sys
import json
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Load environment variables
load_dotenv()

console = Console()

# ─── Configuration ──────────────────────────────────────────────────
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "medgemma"
TOP_K_RESULTS = 4
EXPANDED_K_RESULTS = 8
MAX_GENERATION_RETRIES = 2

# ─── Self-RAG Prompt Templates ─────────────────────────────────────

RETRIEVAL_DECISION_PROMPT = """You are a retrieval decision agent for a Diabetic Retinopathy unified knowledge base.

The knowledge base contains:
- Research papers and clinical guidelines on Diabetic Retinopathy (PDFs covering epidemiology, risk factors, screening, treatment, pathophysiology)
- Patient medical records (patient history, visits, prescriptions, diagnoses)
- Fundus image metadata (retinal photographs classified by DR severity: No DR, Mild, Moderate, Severe, Proliferative)

Given the user's question, decide if it requires retrieving information from the knowledge base.

Answer "yes" if the question is about medical topics, diabetic retinopathy, patient records, clinical guidelines, research papers, fundus images, eye diseases, diabetes complications, treatments, symptoms, diagnosis, prevention, screening, or any related health topic.
Answer "no" if the question is a greeting, casual conversation, thanks, or can be answered without specific knowledge.

Question: {question}

Respond with ONLY a valid JSON object (no other text):
{{"decision": "yes", "reason": "brief explanation"}}"""

RELEVANCE_GRADING_PROMPT = """You are a document relevance grader. Given a user's question and a retrieved document chunk, determine if the chunk contains information relevant to answering the question.

A chunk is relevant if it contains facts, details, or context that would help answer the question, even partially.

Question: {question}

Document chunk:
\"\"\"
{chunk}
\"\"\"

Respond with ONLY a valid JSON object (no other text):
{{"relevant": "yes", "reason": "brief explanation"}}"""

GENERATION_PROMPT = """You are a specialized medical AI assistant for Diabetic Retinopathy with access to a unified knowledge base.

The context below may contain information from different document types:
- Research papers and clinical guidelines on DR (epidemiology, risk factors, screening protocols, treatment options, pathophysiology)
- Patient records with visit histories, prescriptions, and diagnoses
- Fundus image metadata (retinal images classified by DR severity stage with clinical descriptions)

IMPORTANT RULES:
1. Answer questions related to Diabetic Retinopathy, patient records, clinical guidelines, research findings, and fundus imaging.
2. Base your answers primarily on the provided context. If the context doesn't contain enough information, say so.
3. Provide clear, well-structured answers with medical accuracy.
4. Include relevant details like statistics, treatment options, patient history, or diagnostic methods when available.
5. When referring to patient records, present the information clearly and structured.
6. When the context contains fundus image metadata, reference the relevant DR stage and image dataset information.
7. Always remind users to consult healthcare professionals for medical advice.

Context:
{context}

Question: {question}

Answer:"""

NO_RETRIEVAL_PROMPT = """You are a friendly medical AI assistant for Diabetic Retinopathy.
The user's message doesn't require knowledge base retrieval. Respond naturally and helpfully.
If the user is greeting you, greet them back and let them know you can answer questions about:
- Diabetic Retinopathy research papers and clinical guidelines
- Patient medical records
- Fundus image datasets (retinal photographs by DR severity stage)

User message: {question}

Answer:"""

HALLUCINATION_CHECK_PROMPT = """You are a hallucination checker. Your job is to verify whether a generated answer is grounded in the provided source documents.

An answer is "grounded" if all its factual claims can be traced back to the source documents. Minor paraphrasing is acceptable.
An answer is "not grounded" if it contains facts, statistics, or claims that are NOT present in the source documents.

Source documents:
\"\"\"
{context}
\"\"\"

Generated answer:
\"\"\"
{answer}
\"\"\"

Respond with ONLY a valid JSON object (no other text):
{{"grounded": "yes", "reason": "brief explanation"}}"""

ANSWER_QUALITY_PROMPT = """You are an answer quality evaluator. Given a user's question and a generated answer, determine if the answer actually resolves or addresses the user's question adequately.

An answer is "useful" if it directly addresses the question and provides helpful information.
An answer is "not useful" if it is off-topic, too vague, or fails to address the core question.

Question: {question}

Answer:
\"\"\"
{answer}
\"\"\"

Respond with ONLY a valid JSON object (no other text):
{{"useful": "yes", "reason": "brief explanation"}}"""


# ─── Utilities ──────────────────────────────────────────────────────

def parse_json_response(text):
    """Parse JSON from LLM response, handling markdown code blocks."""
    # Try to extract from code blocks first
    code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block:
        text = code_block.group(1)

    # Find JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def check_prerequisites():
    """Verify all prerequisites are met before starting."""
    api_key = "ollama"  # No real key needed for local Ollama

    if not os.path.exists(CHROMA_DB_DIR):
        console.print(Panel(
            "[red bold]Vector database not found![/red bold]\n\n"
            "Run the ingestion script first:\n"
            "  [green]python ingest.py[/green]",
            title="  Database Missing",
            border_style="red",
        ))
        sys.exit(1)

    return api_key


def initialize_components(api_key):
    """Initialize the Self-RAG pipeline components."""
    console.print("[cyan] Loading embedding model...[/cyan]")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    console.print("[cyan] Loading vector store...[/cyan]")
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="diabetic_retinopathy",
    )

    console.print(f"[cyan] Initializing LLM ({LLM_MODEL}) via Ollama...[/cyan]")
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key=api_key,
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=1024,
    )

    return llm, vector_store


def format_sources(source_docs):
    """Format source documents for display with type information."""
    sources = []
    seen = set()
    for doc in source_docs:
        source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
        doc_type = doc.metadata.get("type", "knowledge")
        key = f"{doc_type}:{source_file}"
        if key not in seen:
            seen.add(key)
            sources.append((doc_type, source_file))
    return sources


# ─── Self-RAG Pipeline ─────────────────────────────────────────────

class SelfRAGPipeline:
    """Self-Reflective RAG Pipeline with 5 reasoning stages."""

    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.total_chunks = vector_store._collection.count()

    def _call_llm(self, prompt):
        """Call the LLM and return the response text."""
        response = self.llm.invoke(prompt)
        text = response.content
        # Strip <think>...</think> reasoning blocks from Qwen 3 models
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return text

    # ── Stage 1: Retrieval Decision ──

    def step1_retrieval_decision(self, question):
        """Decide if retrieval from the knowledge base is needed."""
        prompt = RETRIEVAL_DECISION_PROMPT.format(question=question)
        response = self._call_llm(prompt)
        result = parse_json_response(response)

        if result:
            needs_retrieval = result.get("decision", "yes").lower() == "yes"
            reason = result.get("reason", "")
            return needs_retrieval, reason

        return True, "Default to retrieval"

    # ── Stage 2: Retrieve & Grade Relevance ──

    def step2_retrieve_and_grade(self, question, k=TOP_K_RESULTS):
        """Retrieve chunks and grade each one for relevance."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        documents = retriever.invoke(question)

        graded = []
        for doc in documents:
            prompt = RELEVANCE_GRADING_PROMPT.format(
                question=question,
                chunk=doc.page_content,
            )
            response = self._call_llm(prompt)
            result = parse_json_response(response)

            if result:
                relevant = result.get("relevant", "yes").lower() == "yes"
                reason = result.get("reason", "")
            else:
                relevant = True
                reason = "Assumed relevant"

            graded.append((doc, relevant, reason))

        return graded

    # ── Stage 3: Generate Answer ──

    def step3_generate(self, question, relevant_docs, use_retrieval=True):
        """Generate an answer from the relevant documents."""
        if use_retrieval and relevant_docs:
            context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
            prompt = GENERATION_PROMPT.format(context=context, question=question)
        else:
            prompt = NO_RETRIEVAL_PROMPT.format(question=question)

        return self._call_llm(prompt)

    # ── Stage 4: Hallucination Check ──

    def step4_hallucination_check(self, context_docs, answer):
        """Check if the answer is grounded in the source documents."""
        context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
        prompt = HALLUCINATION_CHECK_PROMPT.format(context=context, answer=answer)
        response = self._call_llm(prompt)
        result = parse_json_response(response)

        if result:
            grounded = result.get("grounded", "yes").lower() == "yes"
            reason = result.get("reason", "")
            return grounded, reason

        return True, "Assumed grounded"

    # ── Stage 5: Answer Quality Check ──

    def step5_answer_quality(self, question, answer):
        """Check if the answer resolves the user's question."""
        prompt = ANSWER_QUALITY_PROMPT.format(question=question, answer=answer)
        response = self._call_llm(prompt)
        result = parse_json_response(response)

        if result:
            useful = result.get("useful", "yes").lower() == "yes"
            reason = result.get("reason", "")
            return useful, reason

        return True, "Assumed useful"

    # ── Pipeline Orchestrator ──

    def run(self, question):
        """Run the full Self-RAG pipeline with rich terminal output."""
        console.print()
        console.print("[bold magenta]Self-RAG Pipeline[/bold magenta]")

        # ── Step 1: Retrieval Decision ──────────────────────────
        with console.status("  [dim]├─[/dim] Step 1: Retrieval Decision...", spinner="dots"):
            needs_retrieval, reason = self.step1_retrieval_decision(question)

        if needs_retrieval:
            console.print(
                f"  [dim]├─[/dim] [bold]Step 1: Retrieval Decision[/bold] ─── "
                f"[bold green] Retrieval needed[/bold green] [dim]({reason})[/dim]"
            )
        else:
            console.print(
                f"  [dim]├─[/dim] [bold]Step 1: Retrieval Decision[/bold] ─── "
                f"[bold yellow]No retrieval needed[/bold yellow] [dim]({reason})[/dim]"
            )
            # Generate without retrieval context
            with console.status("  [dim]└─[/dim] Generating answer...", spinner="dots"):
                answer = self.step3_generate(question, [], use_retrieval=False)
            console.print("  [dim]└─[/dim] [bold green] Response generated (no retrieval)[/bold green]")
            console.print()
            return answer, [], []

        # ── Step 2: Retrieve & Grade ────────────────────────────
        with console.status("  [dim]├─[/dim] Step 2: Retrieving and grading chunks...", spinner="dots"):
            graded_docs = self.step2_retrieve_and_grade(question)

        console.print(
            f"  [dim]├─[/dim] [bold]Step 2: Retrieved {len(graded_docs)} chunks, graded relevance:[/bold]"
        )

        relevant_docs = []
        for i, (doc, relevant, grading_reason) in enumerate(graded_docs, 1):
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            is_last = i == len(graded_docs)
            connector = "└─" if is_last else "├─"

            if relevant:
                console.print(
                    f"  [dim]│   {connector}[/dim] Chunk {i} [dim]({source})[/dim] ─── "
                    f"[bold green] Relevant[/bold green] [dim]({grading_reason})[/dim]"
                )
                relevant_docs.append(doc)
            else:
                console.print(
                    f"  [dim]│   {connector}[/dim] Chunk {i} [dim]({source})[/dim] ─── "
                    f"[bold red] Not relevant[/bold red] [dim]({grading_reason})[/dim]"
                )

        # Chunk stats
        console.print(
            f"  [dim]├─[/dim] [bold]Using {len(relevant_docs)} of {len(graded_docs)} "
            f"retrieved chunks ({self.total_chunks} total in store)[/bold]"
        )

        # Fallback if no chunks are relevant
        if not relevant_docs:
            console.print(
                "  [dim]│[/dim]   [yellow]No relevant chunks found, "
                "using all retrieved chunks as fallback[/yellow]"
            )
            relevant_docs = [doc for doc, _, _ in graded_docs]

        all_source_docs = relevant_docs

        # ── Step 3: Generate ────────────────────────────────────
        answer = None
        for attempt in range(MAX_GENERATION_RETRIES + 1):
            with console.status("  [dim]├─[/dim] Step 3: Generating answer...", spinner="dots"):
                answer = self.step3_generate(question, relevant_docs)

            console.print("  [dim]├─[/dim] [bold]Step 3: Answer generated[/bold]")

            # ── Step 4: Hallucination Check ─────────────────────
            with console.status("  [dim]├─[/dim] Step 4: Checking for hallucinations...", spinner="dots"):
                grounded, grounding_reason = self.step4_hallucination_check(relevant_docs, answer)

            if grounded:
                console.print(
                    f"  [dim]├─[/dim] [bold]Step 4: Hallucination check[/bold] ─── "
                    f"[bold green] Answer is grounded[/bold green] [dim]({grounding_reason})[/dim]"
                )
                break
            else:
                if attempt < MAX_GENERATION_RETRIES:
                    console.print(
                        f"  [dim]├─[/dim] [bold]Step 4: Hallucination check[/bold] ─── "
                        f"[bold red] Not grounded[/bold red] [dim]({grounding_reason})[/dim]"
                    )
                    console.print(
                        f"  [dim]│[/dim]   [yellow]Retrying generation "
                        f"(attempt {attempt + 2}/{MAX_GENERATION_RETRIES + 1})...[/yellow]"
                    )
                else:
                    console.print(
                        f"  [dim]├─[/dim] [bold]Step 4: Hallucination check[/bold] ─── "
                        f"[bold yellow]Not fully grounded[/bold yellow] [dim]({grounding_reason})[/dim]"
                    )
                    console.print(
                        "  [dim]│[/dim]   [yellow]Max retries reached, "
                        "proceeding with best answer[/yellow]"
                    )

        # ── Step 5: Answer Quality ──────────────────────────────
        with console.status("  [dim]└─[/dim] Step 5: Evaluating answer quality...", spinner="dots"):
            useful, quality_reason = self.step5_answer_quality(question, answer)

        if useful:
            console.print(
                f"  [dim]└─[/dim] [bold]Step 5: Answer quality[/bold] ─── "
                f"[bold green] Resolves the question[/bold green] [dim]({quality_reason})[/dim]"
            )
        else:
            console.print(
                f"  [dim]├─[/dim] [bold]Step 5: Answer quality[/bold] ─── "
                f"[bold yellow]May not fully resolve[/bold yellow] [dim]({quality_reason})[/dim]"
            )
            # Try with expanded retrieval
            if TOP_K_RESULTS < EXPANDED_K_RESULTS:
                console.print(
                    f"  [dim]│[/dim]   [yellow]Retrying with expanded retrieval "
                    f"(k={EXPANDED_K_RESULTS})...[/yellow]"
                )
                with console.status("  [dim]└─[/dim] Expanding retrieval...", spinner="dots"):
                    expanded_graded = self.step2_retrieve_and_grade(question, k=EXPANDED_K_RESULTS)
                    expanded_relevant = [doc for doc, rel, _ in expanded_graded if rel]
                    if expanded_relevant:
                        answer = self.step3_generate(question, expanded_relevant)
                        all_source_docs = expanded_relevant
                console.print(
                    f"  [dim]└─[/dim] [bold green] Regenerated with "
                    f"{len(expanded_relevant) if expanded_relevant else 0} relevant chunks[/bold green]"
                )

        console.print()
        return answer, all_source_docs, relevant_docs


# ─── Display ────────────────────────────────────────────────────────

def display_banner():
    """Display the application banner."""
    banner = """
Diabetic Retinopathy Self-RAG Assistant"""
    console.print(banner, style="bold magenta")
    console.print()


# ─── Main ───────────────────────────────────────────────────────────

def main():
    display_banner()

    # Check prerequisites
    api_key = check_prerequisites()

    # Initialize components
    with console.status("[bold cyan]Initializing Self-RAG pipeline...[/bold cyan]", spinner="dots"):
        llm, vector_store = initialize_components(api_key)

    # Create the Self-RAG pipeline
    pipeline = SelfRAGPipeline(llm, vector_store)

    console.print(
        f"[bold green] System ready! "
        f"({pipeline.total_chunks} chunks in knowledge base)[/bold green]\n"
    )
    console.print("[dim]─" * 60 + "[/dim]\n")

    while True:
        try:
            question = console.input("[bold cyan]🔹 You:[/bold cyan] ").strip()

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q"):
                console.print(
                    "\n[bold magenta] Goodbye! Take care of your eyes! [/bold magenta]\n"
                )
                break

            if question.lower() == "clear":
                console.clear()
                display_banner()
                continue

            # Run the Self-RAG pipeline
            answer, source_docs, relevant_docs = pipeline.run(question)

            # Display the answer
            console.print(Panel(
                Markdown(answer),
                title="[bold] AI Assistant[/bold]",
    
                padding=(1, 2),
            ))

            # Display sources
            if source_docs:
                sources = format_sources(source_docs)
                source_text = " • ".join(
                    f"[dim]{fname}[/dim] [dim italic]({dtype})[/dim italic]"
                    for dtype, fname in sources
                )
                console.print(f"  [dim] Sources: {source_text}[/dim]")

            console.print(f"\n[dim]─" * 60 + "[/dim]\n")

        except KeyboardInterrupt:
            console.print(
                "\n\n[bold magenta] Goodbye![/bold magenta]\n"
            )
            break
        except Exception as e:
            console.print(f"\n[red] Error: {e}[/red]\n")
            console.print("[dim]Please try again with a different question.[/dim]\n")


if __name__ == "__main__":
    main()
