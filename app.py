#!/usr/bin/env python3
"""Zero-framework web server for the scholarship RAG demo."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from rag import ScholarshipRAG, ingest_pdf, load_chunks

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
PDF = ROOT / "data/source/terms-conditions.pdf"
INDEX = ROOT / "data/index/chunks.json"


def build_engine(reindex: bool = False) -> ScholarshipRAG:
    if reindex or not INDEX.exists():
        chunks = ingest_pdf(PDF, INDEX)
    else:
        chunks = load_chunks(INDEX)
    return ScholarshipRAG(chunks)


class Handler(BaseHTTPRequestHandler):
    engine: ScholarshipRAG

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json({"status": "ok", "chunks": len(self.engine.retriever.chunks)})
            return
        filename = "index.html" if path == "/" else path.lstrip("/")
        candidate = (STATIC / filename).resolve()
        if not str(candidate).startswith(str(STATIC.resolve())) or not candidate.is_file():
            self.send_error(404)
            return
        body = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            question = str(payload.get("question", "")).strip()
            if not question:
                self._json({"error": "Question is required"}, 400)
                return
            method = payload.get("method", "hybrid")
            if method not in {"bm25", "tfidf", "hybrid"}:
                method = "hybrid"
            self._json(self.engine.ask(question, method=method, use_ollama=bool(payload.get("use_ollama", True))))
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[web] {self.address_string()} {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()
    Handler.engine = build_engine(args.reindex)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Scholarship RAG ready at http://{args.host}:{args.port} ({len(Handler.engine.retriever.chunks)} chunks)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

