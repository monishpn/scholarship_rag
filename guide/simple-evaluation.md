# Simple Local Evaluation Guide

This guide explains how to evaluate the scholarship chatbot locally with
`evaluate.py`. It does not use LangSmith, and it does not require Ollama because
the script evaluates retrieval rather than generated answers.

## What is being evaluated?

The evaluation checks whether BM25, TF-IDF, and hybrid retrieval find the
correct section of the scholarship terms and conditions.

The labelled questions are stored in:

```text
evaluation/questions.json
```

Each test case contains:

- A question.
- Whether the question can be answered from the scholarship document.
- The section or sections expected to contain the answer.

The current set contains both answerable scholarship questions and unrelated
questions that the chatbot should reject.

## 1. Open the project directory

### macOS

```bash
cd /path/to/scholarship_rag
```

### Windows Command Prompt

```cmd
cd "C:\path\to\scholarship_rag"
```

### Windows PowerShell

```powershell
cd "C:\path\to\scholarship_rag"
```

Use quotation marks when the directory path contains spaces.

## 2. Create and activate the virtual environment

You only need to create the environment once.

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should normally show `(.venv)` before the
prompt.

## 3. Install the requirements

Run this after activating the environment:

```bash
python -m pip install -r requirements.txt
```

The command is the same on macOS, Windows Command Prompt, and Windows
PowerShell.

## 4. Run the complete evaluation

```bash
python evaluate.py --method all
```

This evaluates all three retrieval options:

- `bm25`
- `tfidf`
- `hybrid`

Ollama does not need to be running for this command.

## Evaluate only one retrieval method

BM25 only:

```bash
python evaluate.py --method bm25
```

TF-IDF only:

```bash
python evaluate.py --method tfidf
```

Hybrid only:

```bash
python evaluate.py --method hybrid
```

## Understanding the metrics

### Hit@1

Hit@1 checks whether the correct section was the first result.

```text
Expected section: 8.131

Rank 1: 8.131  -> Hit@1 = 1
```

If the expected section was ranked second or lower, Hit@1 is `0` for that
question.

### Hit@3

Hit@3 checks whether the correct section appeared anywhere within the first
three results.

```text
Expected section: 8.131

Rank 1: 6.2.1
Rank 2: 8.18
Rank 3: 8.131  -> Hit@3 = 1
```

### MRR@5

MRR means Mean Reciprocal Rank. It rewards the retriever for placing the correct
section close to the top.

| Position of the correct section | Reciprocal rank |
|---:|---:|
| Rank 1 | 1.00 |
| Rank 2 | 0.50 |
| Rank 3 | 0.33 |
| Rank 4 | 0.25 |
| Rank 5 | 0.20 |
| Not in the top five | 0.00 |

The reciprocal ranks for all answerable questions are averaged to produce
MRR@5. A higher value means correct sections usually appear closer to rank one.

### Abstention accuracy

Some test questions cannot be answered from the scholarship PDF. Abstention
accuracy measures how often the system correctly refuses to answer those
questions.

For example, the chatbot should reject a question about the weather because the
scholarship document does not contain weather information.

### False-answer rate

The false-answer rate measures how often the system incorrectly treats an
out-of-scope question as answerable.

A lower false-answer rate is better.

## Understanding the output

The terminal prints one result for every selected retrieval method. A result
looks similar to:

```json
{
  "method": "bm25",
  "questions": 57,
  "answerable_questions": 45,
  "out_of_scope_questions": 12,
  "hit_at_1": 0.8889,
  "hit_at_3": 0.9556,
  "mrr_at_5": 0.9222,
  "abstention_accuracy": 0.9167,
  "false_answer_rate": 0.0833
}
```

Decimal scores can be converted to percentages by multiplying by 100:

```text
0.8889 = 88.89%
0.9556 = 95.56%
0.0833 = 8.33%
```

Do not describe Hit@1, Hit@3, or MRR as the accuracy of the complete chatbot.
They measure retrieval quality, not the correctness of an Ollama-generated
answer.

## Where are the results saved?

The script creates the following files:

```text
evaluation/results/summary.json
evaluation/results/bm25-details.csv
evaluation/results/tfidf-details.csv
evaluation/results/hybrid-details.csv
```

`summary.json` contains the overall metrics. Each CSV contains the result for
every test question, including:

- Whether the question is answerable.
- Whether the system predicted it as answerable.
- The rank of the expected section.
- The top retrieved section.
- The top retrieval score.

The `evaluation/results/` directory is generated automatically and is ignored
by Git.

## Presentation-ready explanation

You can explain the evaluation like this:

> We created a labelled set containing answerable scholarship questions and
> out-of-scope questions. Each answerable question has an expected document
> section. We ran BM25, TF-IDF, and hybrid retrieval and measured Hit@1, Hit@3,
> and MRR@5. We also measured whether the system correctly rejected questions
> that could not be answered from the scholarship document.

## Run the tests as well

Evaluation measures retrieval performance. Unit tests check that important code
behaviour still works:

```bash
python -m unittest discover -s tests -v
```

For a complete local verification, run:

```bash
python -m unittest discover -s tests -v
python evaluate.py --method all
```
