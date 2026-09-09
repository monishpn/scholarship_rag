# How the RMIT Scholarship RAG Chatbot Works

This guide explains the project from first principles for someone who is new to RAG, search systems, language models, and web development. It focuses on what the system does internally and why it was designed this way. For installation and commands, see the project's main `README.md`.

## 1. What this project is

This project is a chatbot that answers questions about the August 2026 RMIT coursework scholarship terms and conditions.

Its trusted knowledge source is:

```text
data/source/terms-conditions.pdf
```

When a student asks a question, the chatbot does not immediately ask an AI model to answer from memory. It first searches the PDF, selects relevant passages, and then uses those passages to construct an answer. It also returns the relevant page and section so the user can inspect the evidence.

The overall flow is:

```text
Scholarship PDF
      ↓
Extract and clean its text
      ↓
Split the text into small chunks
      ↓
Build a searchable in-memory index
      ↓
Student asks a question
      ↓
BM25 and TF-IDF rank the chunks
      ↓
Combine the two rankings
      ↓
Apply scholarship-specific routing rules
      ↓
Check whether the evidence is strong enough
      ↓
Use Ollama or an extractive fallback to answer
      ↓
Return the answer with page and section citations
```

## 2. What RAG means

RAG stands for **Retrieval-Augmented Generation**.

- **Retrieval** means finding relevant information from a knowledge source.
- **Augmented** means adding that information to the prompt sent to a language model.
- **Generation** means using the language model to write a natural-language answer.

A useful analogy is an open-book exam.

A normal chatbot may try to answer using knowledge learned during its training. A RAG chatbot first opens the supplied book, finds the relevant pages, and answers using those pages.

```text
Normal chatbot:
Question → language model's learned knowledge → answer

RAG chatbot:
Question → search trusted documents → relevant passages → language model → answer
```

RAG is useful because scholarship rules can be detailed, institution-specific, and subject to change. The language model should not guess these rules from general knowledge.

RAG reduces hallucination, but it cannot guarantee correctness. The search system can retrieve the wrong passage, and the language model can misunderstand good evidence. That is why this project also has citations, refusal logic, tests, evaluation, and a responsible-use disclaimer.

## 3. The technology stack

The project intentionally uses a small, dependency-light stack.

| Layer | Technology | Responsibility |
|---|---|---|
| Knowledge source | PDF | Stores the official scholarship conditions |
| Backend | Python | Runs ingestion, retrieval, generation, and the HTTP API |
| PDF extraction | `pypdf` | Reads text embedded in the PDF |
| Retrieval | Custom BM25 and TF-IDF | Finds relevant chunks using words |
| Index storage | JSON | Saves processed chunks between runs |
| Answer generation | Optional Ollama model | Rephrases retrieved evidence naturally |
| Offline fallback | Custom extractive answering | Selects sentences when Ollama is unavailable |
| Web server | Python `http.server` | Serves the website and API |
| Frontend | HTML, CSS, JavaScript | Displays the chat, settings, and citations |
| Voice features | Browser speech APIs | Provides speech input and answer read-aloud |
| Testing | Python `unittest` | Tests important behaviors |
| Evaluation | Custom Python script | Measures retrieval and refusal performance |

The project does not currently use React, Flask, Django, LangChain, an OpenAI API, neural embeddings, or a vector database.

## 4. The two major phases

The system has two different phases that should not be confused.

### Phase A: indexing

Indexing happens when the source PDF is first processed or when the application is started with `--reindex`. It converts the PDF into a collection of searchable chunks.

### Phase B: question answering

Question answering happens every time a student sends a message. It searches the already-created chunks, selects evidence, and produces a response.

The PDF is therefore not reparsed for every question.

## 5. Phase A: turning the PDF into an index

### 5.1 Reading the PDF

The `ingest_pdf()` function in `rag.py` opens the PDF with `pypdf.PdfReader` and extracts text one page at a time.

`pypdf` handles PDF decoding; it does not understand scholarship rules. Its job is to return page text as Python strings.

This works best with a digital PDF that contains embedded text. A scanned PDF may contain only page images. Because this project has no OCR system, a scanned PDF may produce little or no usable text.

Complex tables, columns, and unusual page layouts can also be extracted in an incorrect reading order.

### 5.2 Cleaning extracted text

PDF text often includes repeated metadata and awkward line breaks. The `clean_page_text()` function removes lines beginning with patterns such as:

```text
Document:
Author:
Version
Save Date:
Page 2 of 79
```

It also normalizes whitespace and joins words that were hyphenated across line breaks.

The first pages of the current PDF are cover and contents pages. Those pages mention many section titles without containing the actual rules. Indexing them would distort search results, so the ingestion code waits until it finds the body heading `1. Introduction`.

This is one reason the ingestion pipeline is customized for the current document.

### 5.3 Detecting sections

The project expects numbered headings such as:

```text
2.3 Enrolment load
5.3 Taxation
8.131 WIL Grant
```

A regular expression finds these headings and records their section number and title. This gives every chunk useful metadata:

```python
Chunk(
    id="P025-C0084",
    page=25,
    section="8.131",
    title="WIL Grant",
    text="..."
)
```

The metadata is later used for routing and citations.

### 5.4 Chunking

The entire PDF is too large and unfocused to send to a language model for every question. The text is split into smaller passages called **chunks**.

The current defaults are:

```text
Maximum chunk size: 230 words
Overlap:             45 words
Current index:       265 chunks
```

With overlap, the ending of one chunk appears again at the beginning of the next:

```text
Chunk 1: words   1–230
Chunk 2: words 186–415
```

The overlap reduces the chance of losing meaning when an important rule sits at a chunk boundary.

Each chunk receives an ID containing its page and sequence information, such as `P025-C0084`.

### 5.5 Saving the chunks

The processed chunks are stored in:

```text
data/index/chunks.json
```

This is a simple persistent index. On later starts, the program loads this JSON file instead of extracting the PDF again.

The JSON contains text and metadata, not neural vectors.

## 6. Phase B: processing a student's question

Suppose a student asks:

```text
What happens if I do not undertake my WIL placement?
```

The browser sends an HTTP request to `POST /api/chat` containing JSON similar to:

```json
{
  "question": "What happens if I do not undertake my WIL placement?",
  "method": "hybrid",
  "use_ollama": true
}
```

The JavaScript request is in `static/app.js`. The Python endpoint is handled in `app.py`, which passes the question to `ScholarshipRAG.ask()` in `rag.py`.

## 7. Tokenization and stopwords

The first retrieval step is tokenization: converting text into searchable terms.

For example:

```text
Original question:
"What happens if I do not undertake my WIL placement?"

Search tokens:
["happens", "undertake", "wil", "placement"]
```

The program lowercases the text, extracts word-like terms, and removes common words such as `the`, `is`, `what`, `if`, and `my`.

These removed words are called **stopwords**. They appear frequently and usually provide little help when distinguishing one scholarship passage from another.

`rmit` is also ignored because it occurs throughout this particular knowledge base. If it were retained, chunks that repeat the institution's name could receive an artificial advantage.

## 8. BM25 retrieval

BM25 is a word-based ranking algorithm. Calling it “sophisticated keyword matching” means it is smarter than merely counting matching words. It is not an AI model and does not understand language as a human does.

BM25 mainly considers three ideas.

### 8.1 Rare terms are more useful

Suppose the index has 265 chunks:

| Term | Chunks containing it | Approximate usefulness |
|---|---:|---|
| scholarship | 220 | Low |
| student | 180 | Low |
| placement | 15 | Medium |
| WIL | 4 | High |
| Centrelink | 2 | Very high |

`scholarship` is not useful for distinguishing chunks because it appears almost everywhere. `WIL` is much more useful because it narrows the search to a few chunks.

BM25 represents this through inverse document frequency, or IDF. In this context, a “document” means one chunk.

BM25 only knows that a word is statistically uncommon. Describing the word as “meaningful” is informal; the algorithm does not actually understand its meaning.

### 8.2 Repetition helps, but eventually stops helping much

If a query contains `payment`, a passage containing `payment` twice may be more relevant than one containing it once. However, a passage repeating the word 20 times should not automatically be 20 times more relevant.

BM25 gives diminishing benefits to repeated occurrences. This is called **term-frequency saturation**.

### 8.3 Chunk length is normalized

Consider two chunks that each mention `WIL placement` once:

```text
Chunk A:  50 total words
Chunk B: 900 total words
```

The match is a major part of Chunk A but could be incidental in Chunk B. BM25 compares a chunk's length with the average chunk length, preventing unusually long chunks from winning merely because they have more opportunities to contain query words.

“Appropriate document length” therefore means that matching terms are evaluated relative to the chunk's size. It does not mean the algorithm simply prefers the shortest chunk.

The implementation uses the common parameters `k1 = 1.5` and `b = 0.75`:

- `k1` controls how quickly repeated occurrences stop adding value.
- `b` controls how strongly chunk length affects the score.

## 9. TF-IDF retrieval

TF-IDF means **Term Frequency–Inverse Document Frequency**.

- Term frequency measures how often a term occurs in a chunk.
- Inverse document frequency gives more importance to terms appearing in fewer chunks.

The question and each chunk are represented as weighted lists of term values. Cosine similarity compares the direction of these weighted lists. A higher cosine score means the question and chunk emphasize similar words.

Like BM25, TF-IDF is lexical: it compares words rather than using a neural model to understand sentence meaning.

## 10. Hybrid retrieval and Reciprocal Rank Fusion

The project calls its default method **hybrid** because it combines BM25 and TF-IDF.

Suppose they return different rankings:

```text
BM25 ranking               TF-IDF ranking
1. Tax chunk               1. Payments chunk
2. Payments chunk          2. Tax chunk
3. WIL chunk               3. Leave chunk
4. Leave chunk             4. WIL chunk
```

Their raw scores cannot be added fairly because BM25 and TF-IDF use different numerical scales. Instead, the project combines their rank positions using Reciprocal Rank Fusion, or RRF:

```text
RRF score = 1 / (60 + BM25 rank) + 1 / (60 + TF-IDF rank)
```

For a chunk ranked first by BM25 and second by TF-IDF:

```text
1/61 + 1/62 = 0.03252
```

A chunk appearing near the top of both lists usually gets a strong combined result.

### 10.1 Why the constant is 60

The two algorithms are treated equally because both receive one term in the same formula. The number `60` does not create that equality. It controls how sensitive the combined score is to differences in rank.

With a very small constant, first place is much more powerful than second or third place. With a larger constant, differences between positions are gentler, so agreement between retrievers matters more.

Using `6000` would not automatically improve accuracy:

```text
1/6001 ≈ 1/6002 ≈ 1/6100
```

First place and much lower positions would become nearly indistinguishable. Useful ranking information would be washed out.

`60` is a common RRF default, not a universally perfect value. The best value should ideally be selected by comparing candidates such as 10, 20, 40, 60, and 100 on held-out evaluation questions.

## 11. What “lexical” means and what is not implemented

Lexical retrieval is word-based retrieval.

It works well when the question and passage share important vocabulary:

```text
Question: Can I study part-time?
Passage:  Recipients may undertake part-time study.
```

It may struggle with synonyms:

```text
Question: Can I pause my studies?
Passage:  Recipients may apply for a leave of absence.
```

A human recognizes that “pause my studies” and “leave of absence” are related, but the phrases share few useful words.

### 11.1 Neural embeddings

An embedding model converts text into a numerical vector intended to represent approximate meaning:

```text
"Can I pause my studies?"
          ↓
[0.21, -0.73, 0.44, ...]
```

Sentences with similar meanings should receive nearby vectors even when their words differ. Searching these vectors is usually called semantic search.

### 11.2 Vector databases

A vector database stores chunk vectors and efficiently finds vectors close to a question vector. Examples include Qdrant, Pinecone, Weaviate, Milvus, and Chroma. FAISS is commonly used as a vector-search library.

The current project has neither an embedding model nor a vector database. Its hybrid method is:

```text
BM25 lexical search + TF-IDF lexical search
```

It is not the often-used modern combination:

```text
BM25 lexical search + embedding semantic search
```

Ollama may understand paraphrases while writing an answer, but it only sees passages that retrieval has already selected. If lexical retrieval never finds the correct passage, Ollama cannot reliably recover information it was never given.

## 12. Scholarship-specific routing rules

Generic retrieval does not know how this particular PDF is organized. The project adds manually designed signposts called routes.

Examples from `GENERIC_ROUTES` are:

```python
"5.3": {"tax", "taxable", "taxation", "centrelink"}
"2.3": {"part-time", "full-time", "credit", "enrolment", "load"}
"2.6": {"leave", "absence", "hold"}
"5":   {"payment", "payments", "paid", "value"}
"6":   {"performance", "gpa", "grade", "progress", "failed"}
```

If a query contains a cue such as `leave`, the program prioritizes chunks from section `2.6` by adding a large bonus to their combined scores.

For example, ordinary search might initially produce:

```text
1. A named scholarship that mentions leave
2. Section 2.6, the general leave policy
3. Another named scholarship
```

Because the student asked a general leave question, the route boosts section `2.6`, moving it to the top.

Routing does not replace BM25 and TF-IDF. It adjusts their result using developer knowledge about this PDF.

The route is intentionally large enough to prioritize the chosen section:

```python
boost = max(combined) + 1.0
```

These mappings must be reviewed if the PDF's numbering or structure changes.

## 13. How WIL and other named scholarships are routed

There is no manual mapping such as:

```python
"8.131": {"wil"}
```

Instead, named scholarships are detected dynamically from the indexed section titles.

The code examines chunks whose section starts with `8.` because individual scholarships in the current PDF live under section 8.

For the title:

```text
WIL Grant
```

tokenization produces:

```text
wil, grant
```

Generic title words are removed:

```text
scholarship, grant, rmit, support
```

That leaves the distinctive title term:

```text
wil
```

If the question contains `wil`, the program discovers that the matching title belongs to section `8.131` and boosts that section.

Normally, at least two distinctive title terms are required to reduce accidental matches. WIL is explicitly allowed as a one-word special case because it is a distinctive abbreviation:

```python
is_short_distinctive_name = name_terms == {"wil"}
```

Longer names can match automatically when all their distinctive terms occur in the question. Titles that reduce to only one ordinary word may not receive this boost and must rely on BM25 and TF-IDF.

## 14. Concept routes

Some rules are repeated under many individual scholarships instead of appearing once in a general section. Their words may receive weak lexical scores because they occur in numerous chunks.

`CONCEPT_ROUTES` handles selected cases using distinctive query cues. The current example concerns holding another scholarship concurrently. When words such as `simultaneously` or `concurrently` occur, the program finds chunks containing the phrase `any other scholarship`.

For Ollama, `distill_concept_context()` extracts only the matching clauses, removes duplicates, and sends a small set of focused statements. This reduces the chance that a small model blends unrelated conditions from several named scholarships.

## 15. Deciding whether the system should answer

A ranking algorithm always has a first result, even for an unrelated question such as:

```text
What is the weather in Melbourne tomorrow?
```

The first result is merely the least-bad result; that does not make it relevant.

The `is_answerable()` gate therefore requires meaningful evidence. The current rule accepts a question if the top result has:

- a generic route match, or
- a concept route match, or
- a BM25 score of at least `9.0`.

Otherwise, the chatbot returns exactly:

```text
I could not find this information in the scholarship terms and conditions.
```

This behavior is called **abstention**. It is safer than forcing an answer to every question.

The value `9.0` is a calibrated project setting, not a universal BM25 truth. It should be tuned using held-out questions and should be reevaluated when the knowledge base changes.

## 16. Selecting evidence safely

The retriever initially returns up to five ranked results. For an ordinary question, the answer generator receives only highly ranked chunks belonging to the same section as the top result.

This is important because similarly named scholarships can have different rules. Sending unrelated sections together might cause a language model to combine a number or condition from one scholarship with another.

The same evidence is returned to the browser as citations. This keeps the evidence shown to the user aligned with the evidence given to the language model.

## 17. What Ollama is

Ollama is software that downloads, manages, and runs language models locally. Ollama is not itself the language model.

An analogy is:

```text
Ollama       = media player
Llama model  = media file being played
```

The project's default configuration asks Ollama to run:

```text
llama3.2:3b
```

`Llama 3.2` is the model family. `3b` means the model has approximately three billion learned parameters. A parameter is a learned numerical value used during the model's calculations.

Ollama and Llama are broadly similar to ChatGPT in that they can respond to natural-language prompts. However, ChatGPT is a complete hosted application and service, while Ollama is a local model runner that can run different compatible models on the user's computer.

After a model has been downloaded, local use can work without sending prompts to a remote model service. Smaller local models are generally easier to run but may be less capable than large hosted models.

## 18. What Ollama does in this project

Ollama does not:

- read or index the PDF;
- calculate BM25 or TF-IDF;
- find the relevant section;
- decide the retrieval ranking;
- store chunks; or
- create citation metadata.

The Python code performs those tasks.

Ollama's main responsibility is to convert retrieved evidence into a short, friendly response.

For example, formal source wording might say:

```text
Where a recipient fails to undertake the approved WIL placement,
the recipient will be required to return the grant funds to RMIT.
```

The model might phrase it as:

```text
If you do not undertake your approved WIL placement, you will need
to return the grant funds to RMIT.
```

The fact comes from the PDF; the model improves how it is communicated.

## 19. How the language model works at a useful level

At a simplified level, a language model repeatedly predicts the next token. A token may be a word, part of a word, punctuation, or another small text unit.

Given:

```text
The capital of France is
```

the model may assign its highest probability to `Paris`, append it, and then predict the next token. It repeats this process until the response is complete.

During training, the model learns billions of numerical parameters from large amounts of text. These parameters capture language patterns, relationships between concepts, writing styles, and common question-answer structures.

The deeper implementation involves transformers, attention, layers, matrix multiplication, and training through optimization. Those details are not required to understand this project's architecture.

## 20. The prompt sent to Ollama

Python sends an HTTP request to Ollama's local endpoint:

```text
http://localhost:11434/api/generate
```

The request contains:

- the model name;
- a system prompt describing required behavior;
- the student's question;
- the selected PDF passages;
- `stream: false`; and
- `temperature: 0.0`.

The system prompt tells the model to:

- speak like a friendly scholarship adviser;
- use plain language;
- state only facts found in the supplied material;
- avoid unrelated information;
- never invent numbers, dates, percentages, or amounts;
- avoid joining independent conditions into a false relationship; and
- explicitly refuse when the supplied material does not answer the question.

A prompt guides model behavior, but it is not a mathematical guarantee.

Temperature controls variation in token selection. A low value makes output more predictable and conservative. Scholarship guidance should prioritize consistency over creativity, so this project uses `0.0`.

## 21. The extractive fallback

The chatbot still works when Ollama is not installed, is not running, lacks the requested model, or returns an error.

The extractive fallback:

1. Splits the best same-section chunks into sentences.
2. Tokenizes each sentence.
3. Measures overlap with the question terms.
4. Gives preference to focused sentences rather than long sentences with incidental matches.
5. Selects up to three non-duplicate sentences.

This is called **extractive answering** because it selects existing source sentences rather than generating entirely new wording.

```text
Ollama available:   evidence → naturally phrased answer
Ollama unavailable: evidence → selected source sentences
```

## 22. The API response and citations

The backend returns JSON similar to:

```json
{
  "answer": "You will need to return the grant funds to RMIT.",
  "answerable": true,
  "generator": "ollama",
  "sources": [
    {
      "id": "P070-C0240",
      "page": 70,
      "section": "8.131",
      "title": "WIL Grant",
      "citation": "Page 70, Section 8.131 WIL Grant",
      "excerpt": "...",
      "score": 1.0
    }
  ]
}
```

The frontend displays the answer, whether it was grounded, which generator was used, and expandable citations containing the original passage.

The source score exposed by the API is normalized relative to the strongest returned result. It is useful for comparing returned results but should not be interpreted as a perfectly calibrated probability that an answer is correct.

## 23. Web application flow

`app.py` uses Python's standard `ThreadingHTTPServer`, so no web framework is required.

It provides:

- `GET /` for the main interface;
- static CSS and JavaScript files;
- `GET /api/health` for a health check; and
- `POST /api/chat` for questions.

`static/index.html` defines the interface, `static/style.css` controls its appearance, and `static/app.js` handles messages and API calls.

The JavaScript escapes inserted text before displaying it, shows a processing state, renders citations, and handles errors.

The interface can also use browser-provided speech recognition and speech synthesis. Those features depend on browser support and are separate from the RAG engine.

## 24. Evaluation

`evaluation/questions.json` contains labelled questions with expected relevant sections. It includes both answerable scholarship questions and out-of-scope questions.

`evaluate.py` runs BM25, TF-IDF, or hybrid retrieval and calculates:

- **Hit@1:** whether a correct section was the first result;
- **Hit@3:** whether a correct section appeared in the first three results;
- **MRR@5:** how close to the top the first correct result appeared;
- **abstention accuracy:** how often out-of-scope questions were correctly rejected; and
- **false-answer rate:** how often the system tried to answer an out-of-scope question.

Retrieval metrics show whether appropriate evidence was found. They do not prove that every generated answer is correct. Generated answers still require human review.

The current evaluation set is small. Results should not be presented as strong academic evidence until the set is expanded and its relevance labels are independently reviewed.

## 25. Unit tests

`tests/test_rag.py` checks behaviors such as:

- removing stopwords during tokenization;
- cleaning PDF headers;
- retrieving the WIL section;
- rejecting an unrelated question;
- routing a general leave question to section `2.6`; and
- avoiding citations from mixed scholarship sections.

Tests protect known behavior when the code changes, but they only cover the cases that were written. Passing tests do not prove that every possible student question will work.

## 26. What “implemented from scratch” means here

The project implements its central application and retrieval logic directly:

- PDF cleanup;
- section detection;
- chunk creation;
- tokenization and stopwords;
- BM25 scoring;
- TF-IDF weighting;
- cosine similarity;
- reciprocal-rank fusion;
- project-specific routing;
- the answerability gate;
- extractive answering;
- evidence selection and citations;
- the HTTP API;
- the browser interface integration;
- evaluation calculations; and
- unit tests.

It does not implement every underlying technology from mathematical primitives:

- `pypdf` performs PDF decoding and text extraction.
- Ollama runs an already-trained language model.
- The browser supplies speech recognition and speech synthesis.
- Python's standard library supplies the HTTP server, JSON handling, regular expressions, and other utilities.

An accurate description is:

> This is a custom, dependency-light RAG pipeline built from first principles, using `pypdf` for document extraction and optionally Ollama for local answer generation.

## 27. Can any PDF simply be substituted?

Not reliably. Replacing `data/source/terms-conditions.pdf` and running:

```bash
python3 app.py --reindex
```

will make the program attempt to process the new PDF. The general mechanisms—extraction, chunking, BM25, TF-IDF, Ollama, and citations—are reusable. However, the complete application contains assumptions specific to the current RMIT document:

- ingestion waits for the exact body heading `1. Introduction`;
- headings are expected to begin with numbers such as `2.3` or `8.131`;
- header-removal patterns match the current PDF;
- generic routes assume topics are in specific RMIT sections;
- named-scholarship routing assumes individual scholarships are under section 8;
- prompts and refusal messages describe RMIT scholarships;
- tests and evaluation labels refer to the current document; and
- scanned PDFs require OCR, which is not included.

A newer version of the same document with a similar structure will probably work after reindexing, though all section numbers and evaluation cases should be checked. An arbitrary PDF may create zero chunks, incorrect metadata, misleading routes, or poor answers.

To support arbitrary PDFs properly, the project would need configurable or automatic body and heading detection, generic header/footer removal, configurable routes and prompts, OCR support, multi-document handling, and a new evaluated question set for each knowledge base.

## 28. File-by-file map

| File | Purpose |
|---|---|
| `data/source/terms-conditions.pdf` | Trusted knowledge source |
| `data/index/chunks.json` | Generated searchable chunks |
| `rag.py` | Ingestion, retrieval, routing, abstention, and answer generation |
| `app.py` | Web server, static-file serving, and API endpoints |
| `cli.py` | Terminal interface for asking a question |
| `static/index.html` | Chat interface structure |
| `static/style.css` | Interface styling |
| `static/app.js` | Browser behavior, API calls, citations, and voice features |
| `evaluation/questions.json` | Labelled retrieval and out-of-scope questions |
| `evaluate.py` | Evaluation metrics and report generation |
| `tests/test_rag.py` | Unit tests |
| `requirements.txt` | Python dependency version |
| `README.md` | Installation and run instructions |

## 29. One complete example

For the question:

```text
What happens if I do not undertake my WIL placement?
```

the pipeline is:

1. The browser sends the question to `/api/chat`.
2. Python tokenizes it into terms including `undertake`, `wil`, and `placement`.
3. BM25 ranks all 265 chunks.
4. TF-IDF independently ranks the same chunks.
5. RRF combines their ranks.
6. Named-scholarship routing recognizes `WIL Grant` from a section 8 title and boosts section `8.131`.
7. The answerability gate accepts the routed result.
8. Only top evidence from the selected section is prepared for answering.
9. If Ollama is available, Llama receives the question, strict instructions, and this evidence, then writes a short answer.
10. If Ollama is unavailable, the extractive fallback selects the best source sentences.
11. The API returns the answer with the page, section, title, and excerpt.
12. The browser displays the answer and lets the student open its citation.

## 30. The central design idea

The language model is only the final communication layer. It is not the whole chatbot.

```text
Document preparation decides what can be searched.
Retrieval decides what evidence is found.
Routing applies knowledge of this PDF's organization.
Abstention decides whether answering is justified.
Ollama turns selected evidence into natural language.
Citations let the user inspect that evidence.
Evaluation checks whether these decisions work on labelled examples.
```

If retrieval supplies the wrong evidence, even a strong language model can give the wrong answer. For a reliable RAG system, document processing, retrieval, evidence control, refusal behavior, and evaluation are at least as important as the model that writes the final sentences.
