"""Optional LangSmith tracing helpers for the scholarship retriever."""

from __future__ import annotations

from typing import Any

try:
    from langsmith import traceable
except ImportError:  # Keep the local demo usable before dependencies are installed.
    traceable = None


def trace_retrieval(retriever: Any, question: str, top_k: int = 3) -> dict:
    """Deprecated compatibility wrapper; trace one BM25 retrieval."""

    return trace_search(retriever, question, method="bm25", top_k=top_k)


def _serialise_results(results: list[dict]) -> list[dict]:
    return [
        {
            "rank": rank,
            "score": round(result["score"], 6),
            "chunk_id": result["chunk"].id,
            "page": result["chunk"].page,
            "section": result["chunk"].section,
            "title": result["chunk"].title,
            "citation": result["chunk"].citation,
        }
        for rank, result in enumerate(results, start=1)
    ]


def trace_search(retriever: Any, question: str, method: str, top_k: int = 5) -> list[dict]:
    """Run one retriever and return trace-safe ranking data."""

    def retrieve(question: str, method: str, top_k: int) -> list[dict]:
        return _serialise_results(retriever.search(question, top_k=top_k, method=method))

    if traceable is None:
        return retrieve(question, method, top_k)

    traced_retrieve = traceable(
        name=f"scholarship-{method}-retrieval",
        run_type="retriever",
    )(retrieve)
    return traced_retrieve(question, method, top_k)


def trace_answer(engine: Any, question: str, method: str, use_ollama: bool) -> dict:
    """Trace the complete web-chat pipeline without changing its behaviour."""

    def answer(question: str, method: str, use_ollama: bool) -> dict:
        return engine.ask(question, method=method, use_ollama=use_ollama)

    if traceable is None:
        return answer(question, method, use_ollama)
    traced_answer = traceable(name="scholarship-rag-answer", run_type="chain")(answer)
    return traced_answer(question, method, use_ollama)
