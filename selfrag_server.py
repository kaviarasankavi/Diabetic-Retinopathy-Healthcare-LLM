"""
Self-RAG OpenAI-Compatible API Server
======================================
Wraps the Self-RAG pipeline into an OpenAI-compatible /v1/chat/completions
endpoint so it can be used directly inside OpenWebUI as a selectable model.

Usage:
  1. Start this server:  python selfrag_server.py
  2. In OpenWebUI → Settings → Admin → Connections → Add OpenAI connection:
       Base URL: http://localhost:8001/v1
       API Key:  selfrag
  3. Select "self-rag-medgemma" from the model dropdown and chat!
"""

import os
import json
import re
import time
import uuid
import requests as http_requests
from typing import Any, Generator, List, Union

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from security_shield import create_shield, ThreatLevel


class OllamaEmbeddings(Embeddings):
    """LangChain-compatible embeddings using Ollama's /api/embed endpoint.
    Avoids loading torch/transformers locally — sidesteps the Abseil C++ mutex
    deadlock on macOS Apple Silicon entirely."""

    def __init__(self, model: str = "all-minilm", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def _embed(self, texts: List[str]) -> List[List[float]]:
        resp = http_requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

load_dotenv()

# ─── Configuration ──────────────────────────────────────────────────
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
EMBEDDING_MODEL = "all-minilm"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "medgemma"
MODEL_ID = "self-rag-medgemma"
SERVER_PORT = 8001
TOP_K_RESULTS = 4
EXPANDED_K_RESULTS = 8
MAX_GENERATION_RETRIES = 2

# ─── Self-RAG Prompt Templates ─────────────────────────────────────

RETRIEVAL_DECISION_PROMPT = """You are a retrieval decision agent for a Diabetic Retinopathy (DR) knowledge base.

The knowledge base ONLY contains:
- Research papers and clinical guidelines specifically about Diabetic Retinopathy
- Patient medical records related to DR care
- Fundus image metadata (retinal photographs classified by DR severity)

Given the user's question, decide if it requires retrieving from THIS knowledge base.

Answer "yes" ONLY if the question is specifically about:
- Diabetic Retinopathy (DR): symptoms, stages, screening, treatment, risk factors, pathophysiology, epidemiology
- Patient records in the DR database (patient history, visits, prescriptions, diagnoses)
- Fundus images, retinal imaging, DR grading/classification
- Diabetes complications related to the eye (diabetic macular edema, retinal detachment, etc.)
- Clinical guidelines or research papers about DR

Answer "no" if the question is about:
- Other diseases NOT related to Diabetic Retinopathy (e.g., typhoid, cancer, heart disease, flu, COVID, etc.)
- General medical topics not in the knowledge base
- Greetings, casual conversation, or general knowledge questions

When answering "no", the LLM will still answer the question from its own trained medical knowledge.

Question: {question}

Respond with ONLY a valid JSON object (no other text):
{{"decision": "yes" or "no", "reason": "brief explanation"}}"""

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

NO_RETRIEVAL_PROMPT = """You are a friendly and knowledgeable medical AI assistant powered by MedGemma.
You specialize in Diabetic Retinopathy, but you can also answer general medical questions from your trained knowledge.

The user's message doesn't require retrieving from the Diabetic Retinopathy knowledge base.

- If the user is greeting you, greet them back warmly and let them know you specialize in Diabetic Retinopathy but can also help with general medical questions.
- If the user is asking about a medical topic NOT related to DR (e.g., typhoid fever, flu, heart disease), answer the question using your own trained medical knowledge. Provide accurate, helpful, and well-structured medical information.
- Always remind users to consult a healthcare professional for medical advice.

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
    code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block:
        text = code_block.group(1)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None


def clean_llm_response(text):
    """Strip <think>...</think> reasoning blocks from model responses."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# ─── Self-RAG Pipeline ─────────────────────────────────────────────

class SelfRAGPipeline:
    """Self-Reflective RAG Pipeline with 5 reasoning stages."""

    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.total_chunks = vector_store._collection.count()

    def _call_llm(self, prompt, image_urls=None):
        """Call the LLM with a text prompt and optional images."""
        if image_urls:
            from langchain_core.messages import HumanMessage
            content_parts = [{"type": "text", "text": prompt}]
            for url in image_urls:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })
            response = self.llm.invoke([HumanMessage(content=content_parts)])
        else:
            response = self.llm.invoke(prompt)
        return clean_llm_response(response.content)

    def step1_retrieval_decision(self, question):
        prompt = RETRIEVAL_DECISION_PROMPT.format(question=question)
        response = self._call_llm(prompt)
        result = parse_json_response(response)
        if result:
            needs = result.get("decision", "yes").lower() == "yes"
            return needs, result.get("reason", "")
        return True, "Default to retrieval"

    def step2_retrieve_and_grade(self, question, k=TOP_K_RESULTS):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        documents = retriever.invoke(question)
        graded = []
        for doc in documents:
            prompt = RELEVANCE_GRADING_PROMPT.format(
                question=question, chunk=doc.page_content
            )
            response = self._call_llm(prompt)
            result = parse_json_response(response)
            if result:
                relevant = result.get("relevant", "yes").lower() == "yes"
                reason = result.get("reason", "")
            else:
                relevant, reason = True, "Assumed relevant"
            graded.append((doc, relevant, reason))
        return graded

    def step3_generate(self, question, relevant_docs, use_retrieval=True, image_urls=None):
        if use_retrieval and relevant_docs:
            context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
            prompt = GENERATION_PROMPT.format(context=context, question=question)
        else:
            prompt = NO_RETRIEVAL_PROMPT.format(question=question)
        return self._call_llm(prompt, image_urls=image_urls)

    def step4_hallucination_check(self, context_docs, answer):
        context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
        prompt = HALLUCINATION_CHECK_PROMPT.format(context=context, answer=answer)
        response = self._call_llm(prompt)
        result = parse_json_response(response)
        if result:
            grounded = result.get("grounded", "yes").lower() == "yes"
            return grounded, result.get("reason", "")
        return True, "Assumed grounded"

    def step5_answer_quality(self, question, answer):
        prompt = ANSWER_QUALITY_PROMPT.format(question=question, answer=answer)
        response = self._call_llm(prompt)
        result = parse_json_response(response)
        if result:
            useful = result.get("useful", "yes").lower() == "yes"
            return useful, result.get("reason", "")
        return True, "Assumed useful"

    def run_streaming(self, question, image_urls=None) -> Generator[str, None, None]:
        """
        Run the full Self-RAG pipeline, yielding markdown status updates
        for each stage, followed by the final answer.
        """
        yield "## Self-RAG Pipeline\n\n"

        # ── Step 1 ──
        yield "**Step 1: Retrieval Decision** — _thinking..._\n\n"
        needs_retrieval, reason = self.step1_retrieval_decision(question)

        if needs_retrieval:
            yield f"~~Step 1: Retrieval Decision — _thinking..._~~\n\n"
            yield f"**Step 1: Retrieval Decision** — Retrieval needed _{reason}_\n\n"
        else:
            yield f"~~Step 1: Retrieval Decision — _thinking..._~~\n\n"
            yield f"**Step 1: Retrieval Decision** — No retrieval needed _{reason}_\n\n"
            yield "---\n\n"
            answer = self.step3_generate(question, [], use_retrieval=False, image_urls=image_urls)
            yield answer
            return

        # ── Step 2 ──
        yield f"**Step 2: Retrieving & Grading** — _searching {self.total_chunks} chunks..._\n\n"
        graded_docs = self.step2_retrieve_and_grade(question)

        relevant_docs = []
        grade_lines = ""
        for i, (doc, relevant, grading_reason) in enumerate(graded_docs, 1):
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            doc_type = doc.metadata.get("type", "knowledge")
            icon = "[Yes]" if relevant else "[No]"
            status = "Relevant" if relevant else "Not relevant"
            grade_lines += f"  - Chunk {i} `{source}` ({doc_type}) — {icon} {status}: _{grading_reason}_\n"
            if relevant:
                relevant_docs.append(doc)

        yield grade_lines + "\n"

        if not relevant_docs:
            yield "  _No relevant chunks found — using all retrieved chunks as fallback_\n\n"
            relevant_docs = [doc for doc, _, _ in graded_docs]

        yield f"  Using **{len(relevant_docs)}** of {len(graded_docs)} retrieved chunks ({self.total_chunks} total in store)\n\n"

        all_source_docs = relevant_docs

        # ── Steps 3 & 4 (with retry loop) ──
        answer = None
        for attempt in range(MAX_GENERATION_RETRIES + 1):
            yield f"**Step 3: Generating Answer** — _writing (attempt {attempt + 1})..._\n\n"
            answer = self.step3_generate(question, relevant_docs, image_urls=image_urls)

            yield f"**Step 4: Hallucination Check** — _verifying..._\n\n"
            grounded, grounding_reason = self.step4_hallucination_check(relevant_docs, answer)

            if grounded:
                yield f"**Step 4: Hallucination Check** — Grounded _{grounding_reason}_\n\n"
                break
            else:
                if attempt < MAX_GENERATION_RETRIES:
                    yield f"**Step 4: Hallucination Check** — Not grounded _{grounding_reason}_ — _retrying..._\n\n"
                else:
                    yield f"**Step 4: Hallucination Check** — Not fully grounded _{grounding_reason}_ — _max retries reached_\n\n"

        # ── Step 5 ──
        yield f"**Step 5: Answer Quality** — _evaluating..._\n\n"
        useful, quality_reason = self.step5_answer_quality(question, answer)

        if useful:
            yield f"**Step 5: Answer Quality** — Resolves the question _{quality_reason}_\n\n"
        else:
            yield f"**Step 5: Answer Quality** — May not fully resolve _{quality_reason}_\n\n"
            if TOP_K_RESULTS < EXPANDED_K_RESULTS:
                yield f"  _Retrying with expanded retrieval (k={EXPANDED_K_RESULTS})..._\n\n"
                expanded_graded = self.step2_retrieve_and_grade(question, k=EXPANDED_K_RESULTS)
                expanded_relevant = [doc for doc, rel, _ in expanded_graded if rel]
                if expanded_relevant:
                    answer = self.step3_generate(question, expanded_relevant, image_urls=image_urls)
                    all_source_docs = expanded_relevant
                yield f"  _Regenerated with {len(expanded_relevant)} relevant chunks_\n\n"

        # ── Final Answer ──
        yield "---\n\n"
        yield answer
        yield "\n\n---\n\n"

        # ── Sources ──
        seen = set()
        source_parts = []
        for doc in all_source_docs:
            source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
            doc_type = doc.metadata.get("type", "knowledge")
            key = f"{doc_type}:{source_file}"
            if key not in seen:
                seen.add(key)
                source_parts.append(f"`{source_file}` ({doc_type})")
        if source_parts:
            yield "**Sources**: " + " | ".join(source_parts) + "\n"


# ─── OpenAI-Compatible API ──────────────────────────────────────────

app = FastAPI(title="Self-RAG MedGemma API Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
pipeline = None
security_shield = None


class Message(BaseModel):
    role: str
    content: Union[str, List[Any]]


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[Message]
    stream: bool = False
    temperature: float = 0.3
    max_tokens: int = 1024


def extract_multimodal_content(content: Union[str, List[Any]]):
    """Extract text and image URLs from OpenAI-style multimodal content.

    When a user uploads an image via OpenWebUI, the content field becomes a list:
    [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]
    This function normalises both plain-string and list-of-parts formats.

    Returns (text: str, image_urls: list[str])
    """
    if isinstance(content, str):
        return content, []

    text_parts = []
    image_urls = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url:
                    image_urls.append(url)
        elif isinstance(part, str):
            text_parts.append(part)
    return " ".join(text_parts).strip(), image_urls


def make_chunk(content: str, model: str, chat_id: str, finish_reason=None):
    """Create an SSE chunk in OpenAI format."""
    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def stream_response(question: str, model: str, image_urls: list = None) -> Generator[str, None, None]:
    """Stream the Self-RAG pipeline output as OpenAI SSE chunks."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    for text_chunk in pipeline.run_streaming(question, image_urls=image_urls):
        yield make_chunk(text_chunk, model, chat_id)

    yield make_chunk("", model, chat_id, finish_reason="stop")
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "self-rag-local",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, req: Request = None):
    """Handle chat completions (OpenAI-compatible) with security screening."""
    # Extract the latest user message (handles both plain text and multimodal)
    user_message = ""
    image_urls = []
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message, image_urls = extract_multimodal_content(msg.content)
            break

    if not user_message:
        return JSONResponse(
            status_code=400,
            content={"error": "No user message found"},
        )

    # ── Security Shield Gate ──────────────────────────────────
    session_id = "default"
    verdict = security_shield.evaluate(user_message, session_id=session_id)

    if not verdict.allowed:
        # Query blocked — return the refusal message as a normal chat response
        refusal = security_shield.get_refusal_message(verdict.attack_type)
        print(f" BLOCKED [{verdict.attack_type.value}] score={verdict.score:.2f}: {user_message[:80]}...")

        if request.stream:
            def blocked_stream():
                chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                yield make_chunk(refusal, request.model, chat_id)
                yield make_chunk("", request.model, chat_id, finish_reason="stop")
                yield "data: [DONE]\n\n"
            return StreamingResponse(
                blocked_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )
        else:
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": refusal}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

    if verdict.threat_level == ThreatLevel.SUSPICIOUS:
        print(f" SUSPICIOUS [{verdict.attack_type.value}] score={verdict.score:.2f}: {user_message[:80]}...")
        # Use sanitized query if available
        if verdict.sanitized_query and verdict.sanitized_query != user_message:
            user_message = verdict.sanitized_query
            print(f"   → Query sanitized")

    if image_urls:
        print(f" Received {len(image_urls)} image(s) with query: {user_message[:80]}...")

    if request.stream:
        return StreamingResponse(
            stream_response(user_message, request.model, image_urls=image_urls or None),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming: collect the full pipeline output
        full_response = ""
        for chunk in pipeline.run_streaming(user_message, image_urls=image_urls or None):
            full_response += chunk

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_response},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }


@app.get("/health")
async def health():
    audit = security_shield.get_audit_summary() if security_shield else {}
    return {
        "status": "ok",
        "model": MODEL_ID,
        "chunks": pipeline.total_chunks if pipeline else 0,
        "security": audit,
    }


@app.get("/security/audit")
async def security_audit():
    """Return security audit summary."""
    if not security_shield:
        return {"error": "Security shield not initialized"}
    return security_shield.get_audit_summary()


# ─── Startup ────────────────────────────────────────────────────────

def initialize():
    """Initialize all Self-RAG components."""
    global pipeline, security_shield

    print(" Initializing Security Shield (5-layer defense)...")
    security_shield = create_shield(max_queries_per_minute=15)
    print("   ✓ Layer 1: Role-Play Persona Detector")
    print("   ✓ Layer 2: Prompt Injection / Jailbreak Sentinel")
    print("   ✓ Layer 3: Indirect Inference Attack Analyzer")
    print("   ✓ Layer 4: Bulk Data Dump Throttle")
    print("   ✓ Layer 5: Direct Patient Data Extraction Guard")

    print(" Connecting to Ollama embeddings...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print(" Loading vector store...")
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="diabetic_retinopathy",
    )

    print(f" Connecting to Ollama ({OLLAMA_MODEL})...")
    llm = ChatOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        model=OLLAMA_MODEL,
        temperature=0.3,
        max_tokens=1024,
    )

    pipeline = SelfRAGPipeline(llm, vector_store)
    print(f" Self-RAG pipeline ready! ({pipeline.total_chunks} chunks in knowledge base)")
    print(f" Security Shield ACTIVE — all queries screened through 5 defense layers")
    print(f" Server running at http://localhost:{SERVER_PORT}")
    print(f"   Add to OpenWebUI → Settings → Connections → OpenAI:")
    print(f"     Base URL: http://localhost:{SERVER_PORT}/v1")
    print(f"     API Key:  selfrag")
    print(f"   Then select '{MODEL_ID}' from the model dropdown.")


if __name__ == "__main__":
    initialize()
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")
