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
from langsmith import traceable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ask", tags=["RAG — Ask"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    source_filter: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "How does backpropagation work?",
                "top_k": 3,
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

    # Fallback to Mock mode
    logger.info("LLM: Mock Pipeline Mode")
    return None, "fake_llm"


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
    return [
        {
            "chunk": "Backpropagation is the algorithm used to train neural networks.",
            "score": 0.89,
            "source": "ml_basics.txt",
            "chunk_index": 3,
        }
    ][:top_k]


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Document {i} | Source: {chunk['source']} | "
            f"Chunk: {chunk['chunk_index']} | Score: {chunk['score']:.2f}]\n"
            f"{chunk['chunk']}"
        )
    return "\n\n".join(parts)


def calculate_confidence(chunks: list[dict]) -> str:
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




@traceable(name="rag_generation_pipeline")
async def run_rag_pipeline(
    question: str,
    chunks: list[dict],
    llm,
    llm_mode: str,
) -> str:
    """Runs the RAG generation step, dynamically utilizing retrieved chunks."""
    
    if not chunks:
        return "I don't have enough information in the provided documents to answer this."

    # If we are in mock/fake LLM mode, read the context dynamically 
    # from ChromaDB so the answer isn't hardcoded to backpropagation anymore!
    if llm_mode == "fake_llm":
        top_match = chunks[0]
        return (
            f"Based on the provided context in {top_match['source']}: "
            f"{top_match['chunk']}"
        )

    # Real OpenAI Path (if activated via key later)
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        context = format_context(chunks)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer ONLY using the provided context."),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
        ])

        chain = prompt | llm | StrOutputParser()
        answer = await chain.ainvoke({"context": context, "question": question})
        return answer.strip()

    except Exception as e:
        return f"Pipeline error: {e}"


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=AskResponse,
    summary="Ask a question using RAG",
)
async def ask_question(request: AskRequest) -> AskResponse:
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

    mode_summary = f"{retrieval_mode}+{llm_mode}"

    return AskResponse(
        question=request.question,
        answer=answer,
        sources=citations,
        confidence=confidence,
        mode=mode_summary,
    )


@router.get("/info", summary="RAG pipeline status")
async def ask_info() -> dict:
    _, retrieval_mode = get_retrieval_backend()
    _, llm_mode = get_llm()

    return {
        "retrieval": retrieval_mode,
        "llm": llm_mode,
        "status": "fully_operational" if retrieval_mode == "real_retrieval" else "degraded_mock_mode",
    }