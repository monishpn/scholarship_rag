#!/usr/bin/env python3
"""Compare BM25 and TF-IDF retrieval."""
import json
from app import ROOT, build_retriever

def evaluate(method: str) -> dict:
    retriever = build_retriever()
    cases = [c for c in json.loads((ROOT / "evaluation/questions.json").read_text()) if c["answerable"]]
    hit1 = hit3 = 0
    reciprocal_ranks = []
    for case in cases:
        sections = [r["chunk"].section for r in retriever.search(case["question"], 5, method)]
        rank = next((i for i,s in enumerate(sections,1) if s in case["sections"]), 0)
        hit1 += rank == 1
        hit3 += 0 < rank <= 3
        reciprocal_ranks.append(1/rank if rank else 0)
    total = len(cases)
    return {"method":method,"questions":total,"hit_at_1":round(hit1/total,4),"hit_at_3":round(hit3/total,4),"mrr_at_5":round(sum(reciprocal_ranks)/total,4)}

if __name__ == "__main__":
    print(json.dumps([evaluate("bm25"),evaluate("tfidf")],indent=2))
