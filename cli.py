#!/usr/bin/env python3
import argparse
import json
from app import build_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the scholarship document a question")
    parser.add_argument("question")
    parser.add_argument("--method", choices=["bm25", "tfidf", "hybrid"], default="hybrid")
    parser.add_argument("--no-ollama", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_engine().ask(args.question, args.method, not args.no_ollama), indent=2))


if __name__ == "__main__":
    main()

