"""
routers/ask.py
==============
RAG ask endpoint.

Phase 1: FakeListLLM + hardcoded context (works right now).
Phase 3: Real ChromaDB retrieval + real or fake LLM.

The public API contract (request/response shape) never changes.
Only the internal pipeline evolves.
"""

import logging
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ask", tags=["RAG — Ask"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    source_filter: Optional[str] = Field(default=None)
    include_context: bool = Field(default=False)

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "How does backpropagation work?",
                "top_k": 3,
                "include_context": False,
            }
        }
    }


class CitationItem(BaseModel):
    source: str
    chunk_index: int
    score: float
    preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[CitationItem]
    confidence: Literal["high", "medium", "low", "none"]
    mode: str  # "mock" or "real"
    context_chunks: Optional[list[dict]] = None


# ── LLM Builder ───────────────────────────────────────────────────────────────

def get_llm():
    """
    Returns the best available LLM.

    Tries real ChatOpenAI first (needs OPENAI_API_KEY).
    Falls back to FakeListLLM which needs only langchain-community.
    Falls back further to a pure Python mock if langchain isn't installed.
    """
    # Try real OpenAI
    try:
        import os
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key and not api_key.startswith("sk-your"):
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                api_key=api_key,
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("RAG_TEMPERATURE", "0.1")),
            )
            logger.info("LLM: ChatOpenAI (real)")
            return llm, "real_llm"
    except Exception as e:
        logger.warning(f"ChatOpenAI unavailable: {e}")

    # Try FakeListLLM (needs langchain-community)
    try:
        from langchain_community.llms.fake import FakeListLLM
        llm = FakeListLLM(
            responses=[
                (
                    "Based on the provided context, backpropagation is the "
                    "algorithm used to train neural networks. It calculates "
                    "gradients by working backwards through the network, "
                    "allowing the optimizer to adjust weights and minimize loss."
                ),
                (
                    "According to the documents, machine learning enables "
                    "systems to learn from data without explicit programming. "
                    "It is a core subset of artificial intelligence."
                ),
                (
                    "The context explains that FastAPI uses Python type hints "
                    "for automatic validation and documentation generation, "
                    "making it one of the fastest Python web frameworks."
                ),
                (
                    "I don't have enough information in the provided documents "
                    "to answer this question."
                ),
            ]
        )
        logger.info("LLM: FakeListLLM (mock — langchain-community installed)")
        return llm, "fake_llm"
    except Exception as e:
        logger.warning(f"FakeListLLM unavailable: {e}")

    # Pure Python fallback — no langchain at all
    logger.warning("LLM: PurePythonMock (no langchain installed)")
    return None, "python_mock"


def get_retrieval_backend():
    """
    Returns the best available retrieval function.
    Same progressive fallback pattern as search.py.
    """
    try:
        from services.vector_store import search as real_search
        logger.info("Retrieval: ChromaDB (real)")
        return real_search, "real_retrieval"
    except Exception as e:
        logger.warning(f"Real retrieval unavailable ({e}). Using hardcoded context.")
        return hardcoded_retrieval, "mock_retrieval"


def hardcoded_retrieval(
    query: str,
    top_k: int = 3,
    source_filter=None,
) -> list[dict]:
    """
    Returns hardcoded context chunks regardless of the query.
    Phase 1 placeholder — replaced by ChromaDB in Phase 3.
    """
    return [
        {
            "chunk": (
                "Backpropagation is the algorithm used to train neural networks. "
                "It calculates the gradient of the loss function with respect to "
                "each weight by working backwards through the network."
            ),
            "score": 0.89,
            "source": "ml_basics.txt",
            "chunk_index": 3,
        },
        {
            "chunk": (
                "Neural networks consist of layers of interconnected nodes. "
                "Each connection has a weight that is adjusted during training "
                "to minimize the loss function."
            ),
            "score": 0.82,
            "source": "ml_basics.txt",
            "chunk_index": 4,
        },
        {
            "chunk": (
                "Machine learning models learn patterns from training data. "
                "Overfitting occurs when the model memorises training data "
                "rather than generalising to new examples."
            ),
            "score": 0.74,
            "source": "ml_basics.txt",
            "chunk_index": 5,
        },
    ][:top_k]


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block.
    Used identically in mock and real pipelines.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Document {i} | Source: {chunk['source']} | "
            f"Chunk: {chunk['chunk_index']} | Score: {chunk['score']:.2f}]\n"
            f"{chunk['chunk']}"
        )
    return "\n\n".join(parts)


def calculate_confidence(chunks: list[dict]) -> str:
    """Derive confidence from retrieval scores."""
    if not chunks:
        return "none"
    best = chunks[0]["score"]
    avg = sum(c["score"] for c in chunks) / len(chunks)
    if best >= 0.75 and avg >= 0.60:
        return "high"
    elif best >= 0.55 and avg >= 0.45:
        return "medium"
    elif best >= 0.25:
        return "low"
    return "none"


async def run_rag_pipeline(
    question: str,
    chunks: list[dict],
    llm,
    llm_mode: str,
) -> str:
    """
    Runs the RAG generation step.

    Handles three LLM modes:
    - real_llm / fake_llm: uses LangChain LCEL chain
    - python_mock: pure Python string response (no langchain needed)
    """
    context = format_context(chunks)

    if llm_mode == "python_mock":
        # Pure Python fallback — no LangChain at all
        return (
            f"[MOCK ANSWER — install langchain for real generation]\n\n"
            f"Question: {question}\n\n"
            f"Context available:\n{context[:300]}..."
        )

    # LangChain path (works for both real and fake LLM)
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a precise question-answering assistant.
Answer ONLY using the provided context.
If the context doesn't contain the answer, say:
"I don't have enough information in the provided documents to answer this."
Never use outside knowledge.""",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            ),
        ])

        chain = prompt | llm | StrOutputParser()

        # Use async invoke for FastAPI compatibility
        answer = await chain.ainvoke({
            "context": context,
            "question": question,
        })
        return answer.strip()

    except Exception as e:
        logger.error(f"LangChain chain failed: {e}")
        return (
            f"Pipeline error: {e}. "
            f"Context was retrieved but generation failed."
        )


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=AskResponse,
    summary="Ask a question using RAG",
)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Answer a question using Retrieval-Augmented Generation.

    **Current mode depends on installed packages:**
    - No ML packages: mock retrieval + python mock LLM
    - langchain only: mock retrieval + FakeListLLM
    - All packages + ingest run: real ChromaDB + real/fake LLM

    The `mode` field tells you exactly which combination answered.
    """
    retrieval_fn, retrieval_mode = get_retrieval_backend()
    llm, llm_mode = get_llm()

    # Step 1: Retrieve context
    chunks = retrieval_fn(
        query=request.question,
        top_k=request.top_k,
        source_filter=request.source_filter,
    )

    # Step 2: Calculate confidence
    confidence = calculate_confidence(chunks)

    # Step 3: Generate answer
    answer = await run_rag_pipeline(
        question=request.question,
        chunks=chunks,
        llm=llm,
        llm_mode=llm_mode,
    )

    # Step 4: Build citations
    citations = [
        CitationItem(
            source=c["source"],
            chunk_index=c["chunk_index"],
            score=c["score"],
            preview=c["chunk"][:150] + "..." if len(c["chunk"]) > 150 else c["chunk"],
        )
        for c in chunks
    ]

    # Step 5: Build response
    context_chunks = None
    if request.include_context:
        context_chunks = chunks

    mode_summary = f"{retrieval_mode}+{llm_mode}"

    return AskResponse(
        question=request.question,
        answer=answer,
        sources=citations,
        confidence=confidence,
        mode=mode_summary,
        context_chunks=context_chunks,
    )


@router.get("/info", summary="RAG pipeline status")
async def ask_info() -> dict:
    """Shows which components are active (mock vs real)."""
    _, retrieval_mode = get_retrieval_backend()
    _, llm_mode = get_llm()

    return {
        "retrieval": retrieval_mode,
        "llm": llm_mode,
        "status": (
            "fully_operational"
            if retrieval_mode == "real_retrieval" and "real" in llm_mode
            else "degraded_mock_mode"
        ),
        "next_step": (
            "Run python ingest.py to activate real retrieval"
            if retrieval_mode != "real_retrieval"
            else "Set OPENAI_API_KEY to activate real LLM"
        ),
    }