# Scholarship Retrieval Milestone

This branch demonstrates PDF ingestion, tokenization, and lexical retrieval. It intentionally contains no LLM or answer-generation stage.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py --reindex
```

Enter questions at `Question >`. The terminal shows the top three BM25 chunks and top three TF-IDF chunks with their scores. Type `exit` to stop. On later runs, use `python app.py`.

## Verify

```bash
python -m unittest discover -s tests -v
python evaluate.py
```
