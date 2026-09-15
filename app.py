#!/usr/bin/env python3
"""Interactive terminal demonstration of BM25 and TF-IDF retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag import LexicalRetriever, ingest_pdf, load_chunks

ROOT = Path(__file__).resolve().parent
PDF = ROOT / "data/source/terms-conditions.pdf"
INDEX = ROOT / "data/index/chunks.json"


def build_retriever(reindex: bool = False) -> LexicalRetriever:
    chunks = ingest_pdf(PDF, INDEX) if reindex or not INDEX.exists() else load_chunks(INDEX)
    return LexicalRetriever(chunks)


def print_results(label: str, results: list[dict]) -> None:
    print(f"\n{'=' * 72}")
    print(label)
    print("=" * 72)
    for result in results:
        print(f"#{result['rank']}  Score: {result['score']:.6f}")
        print(f"    Chunk: {result['chunk_id']}")
        print(f"    {result['citation']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BM25 and TF-IDF scholarship retrieval")
    parser.add_argument("--reindex", action="store_true", help="rebuild chunks from the PDF")
    args = parser.parse_args()

    retriever = build_retriever(args.reindex)
    print(f"\nScholarship retrieval demo loaded {len(retriever.chunks)} chunks.")
    print("Enter a question to compare BM25 and TF-IDF.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if question.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        if not question:
            continue

        comparison = retriever.compare(question)
        print(f"\nTokens: {comparison['tokens']}")
        print_results("BM25 — top 3 chunks", comparison["bm25"])
        print_results("TF-IDF — top 3 chunks", comparison["tfidf"])
        same = comparison["bm25"][0]["chunk_id"] == comparison["tfidf"][0]["chunk_id"]
        print(f"\nAgreement: {'YES — both selected the same chunk' if same else 'NO — they selected different chunks'}\n")


if __name__ == "__main__":
    main()
