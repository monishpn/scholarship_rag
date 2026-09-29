#!/usr/bin/env python3
"""Run the labelled retrieval dataset as a LangSmith experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from langsmith import Client, evaluate

from app import ROOT, build_engine
from langsmith_support import trace_search

RESULTS_DIR = ROOT / "evaluation/results"


def _rank(outputs: dict, reference_outputs: dict, method: str) -> int:
    expected = set(reference_outputs["sections"])
    return next(
        (
            index
            for index, result in enumerate(outputs[method], start=1)
            if result["section"] in expected
        ),
        0,
    )


def make_evaluator(method: str, metric: str):
    """Create a LangSmith code evaluator for one retrieval metric."""

    def evaluator(outputs: dict, reference_outputs: dict) -> dict:
        rank = _rank(outputs, reference_outputs, method)
        if metric == "hit_at_1":
            score = int(rank == 1)
        elif metric == "hit_at_3":
            score = int(0 < rank <= 3)
        elif metric == "reciprocal_rank":
            score = 1 / rank if rank else 0
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return {"key": f"{method}_{metric}", "score": score}

    evaluator.__name__ = f"{method}_{metric}"
    return evaluator


def score_outputs(outputs: dict, reference_outputs: dict) -> dict[str, float]:
    """Calculate every retrieval metric locally from one question's outputs."""
    scores = {}
    for method in ("bm25", "tfidf"):
        rank = _rank(outputs, reference_outputs, method)
        scores[f"{method}_rank"] = rank
        scores[f"{method}_hit_at_1"] = int(rank == 1)
        scores[f"{method}_hit_at_3"] = int(0 < rank <= 3)
        scores[f"{method}_reciprocal_rank"] = 1 / rank if rank else 0
    return scores


def retrieve_outputs(retriever, question: str, top_k: int = 5, traced: bool = False) -> dict:
    """Return comparable BM25 and TF-IDF rankings from the current RAG engine."""
    if traced:
        return {
            method: trace_search(retriever, question, method=method, top_k=top_k)
            for method in ("bm25", "tfidf")
        }

    def serialise(method: str) -> list[dict]:
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
            for rank, result in enumerate(
                retriever.search(question, top_k=top_k, method=method), start=1
            )
        ]

    return {method: serialise(method) for method in ("bm25", "tfidf")}


def load_cases() -> list[dict]:
    cases = json.loads((ROOT / "evaluation/questions.json").read_text(encoding="utf-8"))
    return [case for case in cases if case["answerable"]]


def dataset_name_for(cases: list[dict]) -> str:
    """Version the remote dataset whenever questions or labels change."""
    content = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(content).hexdigest()[:8]
    return f"scholarship-rag-retrieval-{fingerprint}"


def run_local_preflight(retriever, cases: list[dict]) -> tuple[list[dict], dict]:
    """Produce a complete local result set before contacting LangSmith."""
    rows = []
    for case in cases:
        outputs = retrieve_outputs(retriever, case["question"], top_k=5)
        scores = score_outputs(outputs, {"sections": case["sections"]})
        rows.append({
            "id": case["id"],
            "question": case["question"],
            "expected_sections": ", ".join(case["sections"]),
            "bm25_top_section": outputs["bm25"][0]["section"],
            "tfidf_top_section": outputs["tfidf"][0]["section"],
            **scores,
        })

    summary = {"questions": len(rows)}
    for method in ("bm25", "tfidf"):
        summary[method] = {
            "hit_at_1": sum(row[f"{method}_hit_at_1"] for row in rows) / len(rows),
            "hit_at_3": sum(row[f"{method}_hit_at_3"] for row in rows) / len(rows),
            "mrr_at_5": sum(row[f"{method}_reciprocal_rank"] for row in rows) / len(rows),
        }
    return rows, summary


def save_local_results(rows: list[dict], summary: dict, experiment: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(character if character.isalnum() or character in "-_" else "-" for character in experiment)
    csv_path = RESULTS_DIR / f"{safe_name}.csv"
    summary_path = RESULTS_DIR / f"{safe_name}-summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, summary_path


def print_summary(summary: dict) -> None:
    print("\nLOCAL PRE-FLIGHT (complete dataset)")
    print("-" * 63)
    print(f"{'Retriever':<12}{'Hit@1':>12}{'Hit@3':>12}{'MRR@5':>12}")
    for method in ("bm25", "tfidf"):
        values = summary[method]
        print(
            f"{method.upper():<12}"
            f"{values['hit_at_1']:>11.2%}"
            f"{values['hit_at_3']:>12.2%}"
            f"{values['mrr_at_5']:>12.2%}"
        )
    print(f"\nQuestions evaluated: {summary['questions']}")


def ensure_dataset(client: Client, dataset_name: str, cases: list[dict]) -> None:
    """Create the labelled LangSmith dataset once; never overwrite remote data."""
    if client.has_dataset(dataset_name=dataset_name):
        print(f"Using existing LangSmith dataset: {dataset_name}")
        return

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Scholarship questions labelled with the document section that answers them.",
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": case["question"]},
                "outputs": {"sections": case["sections"]},
                "metadata": {"case_id": case["id"]},
            }
            for case in cases
        ],
    )
    print(f"Created LangSmith dataset '{dataset_name}' with {len(cases)} examples.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate scholarship retrieval in LangSmith")
    parser.add_argument("--dataset", help="LangSmith dataset name (default: content-versioned name)")
    parser.add_argument("--experiment", default="bm25-vs-tfidf", help="experiment name prefix")
    parser.add_argument("--local-only", action="store_true", help="calculate complete local results without uploading")
    args = parser.parse_args()

    cases = load_cases()
    retriever = build_engine().retriever
    rows, summary = run_local_preflight(retriever, cases)
    csv_path, summary_path = save_local_results(rows, summary, args.experiment)
    print_summary(summary)
    print(f"Local detail:  {csv_path}")
    print(f"Local summary: {summary_path}")

    if args.local_only:
        print("\nLocal-only evaluation complete; nothing was uploaded to LangSmith.")
        return
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit(
            "\nLocal evaluation is complete. Set LANGSMITH_API_KEY to upload it, "
            "or use --local-only to suppress this message."
        )

    client = Client()
    dataset_name = args.dataset or dataset_name_for(cases)
    ensure_dataset(client, dataset_name, cases)

    def target(inputs: dict) -> dict:
        return retrieve_outputs(retriever, inputs["question"], top_k=5, traced=True)

    evaluators = [
        make_evaluator(method, metric)
        for method in ("bm25", "tfidf")
        for metric in ("hit_at_1", "hit_at_3", "reciprocal_rank")
    ]
    results = evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=args.experiment,
        metadata={"retrievers": ["bm25", "tfidf"], "top_k": 5},
        client=client,
        max_concurrency=1,
        blocking=True,
    )
    results.wait()
    client.flush()
    print(f"\nLangSmith experiment: {results.experiment_name}")
    print(f"LangSmith URL: {results.url}")
    print("All local metrics were calculated before upload; use them as the completeness check.")


if __name__ == "__main__":
    main()
