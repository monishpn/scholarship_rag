# RMIT Scholarship RAG

An assignment-ready, document-grounded chatbot. Its evaluation approach (BM25 baseline, Hit@k/MRR,
abstention) is inspired by Walert's methodology, but this is a standalone project with no shared
code or dependency on it.

## Included

- August 2026 scholarship terms as the sole knowledge base
- Page- and section-aware PDF chunking
- BM25 baseline, TF-IDF retrieval, and reciprocal-rank-fused hybrid retrieval
- Out-of-knowledge-base detection
- Optional local Ollama answer generation and an offline extractive fallback
- Browser UI with inspectable citations
- Versioned evaluation set, Hit@k, MRR, and abstention metrics
- Unit tests and Docker configuration

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py --reindex
```

Open <http://localhost:8000>. The first start creates `data/index/chunks.json`; later starts reuse it.

To use Ollama, start it separately. The default model is `llama3.2:3b`; no model is downloaded automatically. Without Ollama, the app returns extractive answers.

```bash
OLLAMA_MODEL=llama3.2:3b .venv/bin/python app.py
```

## Test and evaluate

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python evaluate.py --method all
```

Results are written to `evaluation/results/`. Before citing results academically, expand to at least 50 questions and have relevance judgments independently checked by another team member.

## Docker

With Docker Desktop running:

```bash
docker compose up --build
```

Ollama stays on the host, keeping the image small. The chatbot works without it.

## API

```bash
curl -s http://localhost:8000/api/health
curl -s http://localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"question":"What happens if I do not undertake my WIL placement?","method":"hybrid","use_ollama":false}'
```

## Evaluation interpretation

BM25 is the Walert-style lexical baseline. TF-IDF supplies another independently scored representation, while hybrid retrieval combines both rankings. TF-IDF is deliberately not presented as a neural semantic retriever. Sentence embeddings can be added later as another controlled experiment.

## Responsible-use limitations

- This prototype provides guidance, not official eligibility decisions.
- Re-index whenever the source document changes.
- Retrieval metrics alone do not prove answer correctness; conduct human review.
- Do not send personal, health, financial, or visa data to remote models.
- Keep citations visible and abstain when evidence is insufficient.
# scholarship_rag
