#!/usr/bin/env python3
"""Evaluate retrieval and out-of-knowledge-base handling."""

import argparse
import csv
import json
from app import ROOT, build_engine
from rag import is_answerable


def evaluate(method):
    engine = build_engine()
    cases = json.loads((ROOT / "evaluation/questions.json").read_text())
    rows, reciprocal_ranks = [], []
    answerable_count = abstain_count = correct_abstain = false_answer = hit1 = hit3 = 0
    for case in cases:
        results = engine.retriever.search(case["question"], top_k=5, method=method)
        predicted = is_answerable(results)
        sections = [r["chunk"].section for r in results]
        rank = next((i for i, section in enumerate(sections, 1) if section in case["sections"]), 0)
        if case["answerable"]:
            answerable_count += 1
            hit1 += rank == 1
            hit3 += 0 < rank <= 3
            reciprocal_ranks.append(1 / rank if rank else 0)
        else:
            abstain_count += 1
            correct_abstain += not predicted
            false_answer += predicted
        rows.append({"id":case["id"], "method":method, "answerable":case["answerable"],
                     "predicted_answerable":predicted, "relevant_rank":rank,
                     "top_section":sections[0] if sections else "",
                     "top_score":round(results[0]["score"], 6) if results else 0})
    return {"metrics":{"method":method, "questions":len(cases), "answerable_questions":answerable_count,
            "out_of_scope_questions":abstain_count, "hit_at_1":round(hit1/answerable_count, 4),
            "hit_at_3":round(hit3/answerable_count, 4), "mrr_at_5":round(sum(reciprocal_ranks)/answerable_count, 4),
            "abstention_accuracy":round(correct_abstain/abstain_count, 4),
            "false_answer_rate":round(false_answer/abstain_count, 4)}, "rows":rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bm25","tfidf","hybrid","all"], default="all")
    args = parser.parse_args()
    methods = ["bm25","tfidf","hybrid"] if args.method == "all" else [args.method]
    output_dir = ROOT / "evaluation/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for method in methods:
        report = evaluate(method)
        summaries.append(report["metrics"])
        with (output_dir / f"{method}-details.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=report["rows"][0].keys())
            writer.writeheader(); writer.writerows(report["rows"])
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
