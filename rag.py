"""Small, dependency-light RAG engine for the RMIT scholarship document."""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
SECTION_RE = re.compile(r"(?m)^(\d+(?:\.\d+)*)(?:\.)?\s+([^\n]{3,120})$")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "how", "i", "if", "in", "is", "it", "my", "of", "on", "or", "that", "the",
    "this", "to", "what", "when", "where", "which", "who", "will", "with", "you", "your",
    # Appears in almost every chunk's body and in ~8 scholarship titles specifically (e.g.
    # "RMIT Accommodation Support Scholarship"). Titles are weighted 2x, so those particular
    # chunks get an artificial edge on this near-universal word for nearly every real query.
    "rmit",
}
# Facts repeated near-identically across dozens of per-scholarship clauses rather than stated
# once in a general section. Lexical scores for these stay low everywhere, so route on rare,
# distinctive query words instead of a section number, and match any chunk containing the phrase.
CONCEPT_ROUTES = {
    "any other scholarship": {"simultaneously", "concurrently", "concurrent"},
}
GENERIC_ROUTES = {
    "5.3": {"tax", "taxable", "taxation", "centrelink"},
    "4.2.1": {"change", "changes", "changing", "transfer", "transferring", "defer", "deferring", "deferral", "postpone"},
    "2.3": {"part-time", "parttime", "full-time", "fulltime", "credit", "credits", "enrolment", "load"},
    "2.6": {"leave", "absence", "hold"},
    "3": {"email", "correspondence", "message", "contact", "respond", "responding", "response"},
    "5": {"payment", "payments", "paid", "value"},
    # "academic" alone was too broad (matched unrelated queries like "next academic semester");
    # the more specific words below already cover genuine GPA/performance questions.
    "6": {"retain", "retaining", "performance", "gpa", "grade", "grades",
          "progress", "fail", "failing", "failed", "unsatisfactory"},
    "6.2.1": {"terminate", "terminating", "terminated", "termination", "cancel", "cancelled", "revoke", "revoked"},
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def clean_page_text(text: str) -> str:
    lines = []
    for line in text.replace("\u00a0", " ").splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            lines.append("")
            continue
        if re.match(r"^(Document:|Author:|Version |Save Date:|Page \d+ of \d+)", line):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass(frozen=True)
class Chunk:
    id: str
    page: int
    section: str
    title: str
    text: str

    @property
    def citation(self) -> str:
        label = self.section + (f" {self.title}" if self.title else "")
        return f"Page {self.page}, Section {label}".strip()


def _window(words: list[str], size: int, overlap: int) -> Iterable[list[str]]:
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if piece:
            yield piece
        if start + size >= len(words):
            break


def ingest_pdf(pdf_path: Path, output_path: Path, chunk_words: int = 230, overlap: int = 45) -> list[Chunk]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    reader = PdfReader(str(pdf_path))
    chunks: list[Chunk] = []
    current_section, current_title = "", "General conditions"
    body_started = False
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_page_text(page.extract_text() or "")
        # The first three pages are cover/contents pages containing every section
        # title; indexing them badly distorts both retrieval and section metadata.
        if not body_started:
            body_started = bool(re.search(r"(?m)^1\.\s+Introduction\s*$", text) and "Contents" not in text)
        if not body_started:
            continue
        if not text:
            continue
        # A payment-schedule table row like "1 year (2 semesters) 1 2 1.5 years" matches the
        # heading pattern too. Real section titles never contain multiple bare numbers, so use
        # that to reject table rows without relying on a fragile document-wide ordering check.
        matches = [
            m for m in SECTION_RE.finditer(text)
            if len(re.findall(r"\d+(?:\.\d+)?", m.group(2))) < 2
        ]
        segments: list[tuple[str, str, str]] = []
        if matches:
            prefix = text[: matches[0].start()].strip()
            if prefix:
                segments.append((current_section, current_title, prefix))
            for idx, match in enumerate(matches):
                current_section, current_title = match.group(1), match.group(2).strip()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                segments.append((current_section, current_title, text[match.end() : end].strip()))
        else:
            segments.append((current_section, current_title, text))

        for section, title, segment in segments:
            words = segment.split()
            if not words:
                continue
            heading = f"Section {section}: {title}" if section else title
            for piece in _window(words, chunk_words, overlap):
                chunk_id = f"P{page_number:03d}-C{len(chunks) + 1:04d}"
                chunks.append(Chunk(chunk_id, page_number, section, title, f"{heading}\n{' '.join(piece)}"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(c) for c in chunks], indent=2), encoding="utf-8")
    return chunks


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


class HybridRetriever:
    """BM25 plus TF-IDF cosine, fused with reciprocal-rank fusion."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.docs = [tokenize(f"{c.title} {c.title} {c.text}") for c in chunks]
        self.term_freqs = [Counter(doc) for doc in self.docs]
        self.lengths = [len(doc) for doc in self.docs]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency: Counter[str] = Counter()
        for doc in self.docs:
            document_frequency.update(set(doc))
        n = max(1, len(self.docs))
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in document_frequency.items()}

    def _bm25(self, query: list[str]) -> list[float]:
        k1, b = 1.5, 0.75
        scores = []
        for tf, length in zip(self.term_freqs, self.lengths):
            score = 0.0
            for term in query:
                freq = tf.get(term, 0)
                if freq:
                    denom = freq + k1 * (1 - b + b * length / max(1, self.avg_length))
                    score += self.idf.get(term, 0.0) * freq * (k1 + 1) / denom
            scores.append(score)
        return scores

    def _tfidf(self, query: list[str]) -> list[float]:
        qtf = Counter(query)
        qvec = {term: freq * self.idf.get(term, 0.0) for term, freq in qtf.items()}
        qnorm = math.sqrt(sum(value * value for value in qvec.values())) or 1.0
        scores = []
        for tf in self.term_freqs:
            dot = sum(qvec.get(term, 0.0) * freq * self.idf.get(term, 0.0) for term, freq in tf.items())
            dnorm = math.sqrt(sum((freq * self.idf.get(term, 0.0)) ** 2 for term, freq in tf.items())) or 1.0
            scores.append(dot / (qnorm * dnorm))
        return scores

    @staticmethod
    def _ranks(scores: list[float]) -> dict[int, int]:
        return {doc_id: rank for rank, doc_id in enumerate(sorted(range(len(scores)), key=scores.__getitem__, reverse=True), start=1)}

    def search(self, question: str, top_k: int = 5, method: str = "hybrid") -> list[dict]:
        query = tokenize(question)
        if not query:
            return []
        bm25 = self._bm25(query)
        tfidf = self._tfidf(query)
        if method == "bm25":
            combined = bm25
        elif method == "tfidf":
            combined = tfidf
        else:
            bm_ranks, tf_ranks = self._ranks(bm25), self._ranks(tfidf)
            combined = [1 / (60 + bm_ranks[i]) + 1 / (60 + tf_ranks[i]) for i in range(len(self.chunks))]

        query_set = set(query)
        specific_sections = set()
        for chunk in self.chunks:
            if not chunk.section.startswith("8."):
                continue
            name_terms = set(tokenize(chunk.title)) - {"scholarship", "grant", "rmit", "support"}
            is_short_distinctive_name = name_terms == {"wil"}
            if name_terms.issubset(query_set) and (len(name_terms) >= 2 or is_short_distinctive_name):
                specific_sections.add(chunk.section)
        routed_sections = set()
        if specific_sections:
            routed_sections = specific_sections
        else:
            for section, cues in GENERIC_ROUTES.items():
                if query_set & cues:
                    routed_sections = {section}
                    break
        if routed_sections:
            boost = max(combined, default=0.0) + 1.0
            for i, chunk in enumerate(self.chunks):
                if chunk.section in routed_sections:
                    combined[i] += boost

        concept_indices = set()
        for needle, cues in CONCEPT_ROUTES.items():
            if query_set & cues:
                for i, chunk in enumerate(self.chunks):
                    if needle in chunk.text.lower():
                        concept_indices.add(i)
        if concept_indices:
            boost = max(combined, default=0.0) + 1.0
            for i in concept_indices:
                combined[i] += boost

        order = sorted(range(len(combined)), key=combined.__getitem__, reverse=True)[:top_k]
        maximum = max((combined[i] for i in order), default=1.0) or 1.0
        return [
            {"chunk": self.chunks[i], "score": combined[i], "confidence": combined[i] / maximum,
             "route_match": self.chunks[i].section in routed_sections,
             "concept_match": i in concept_indices,
             "bm25_score": bm25[i], "tfidf_score": tfidf[i]}
            for i in order if combined[i] > 0
        ]


def extractive_answer(question: str, results: list[dict]) -> str:
    query = set(tokenize(question))
    candidates = []
    # Avoid blending similarly named scholarships. Continue to a second chunk
    # only when it belongs to the same section as the strongest result.
    top_section = results[0]["chunk"].section if results else ""
    supporting = [result for result in results if result["chunk"].section == top_section][:2]
    for result in supporting:
        chunk: Chunk = result["chunk"]
        for sentence in SENTENCE_RE.split(chunk.text.replace("\n", " ")):
            terms = set(tokenize(sentence))
            overlap = len(query & terms)
            if overlap:
                candidates.append((overlap / math.sqrt(max(1, len(terms))), sentence.strip()))
    chosen = []
    seen = set()
    for _, sentence in sorted(candidates, reverse=True):
        key = sentence.lower()
        if key not in seen and len(sentence) > 25:
            chosen.append(sentence)
            seen.add(key)
        if len(chosen) == 3:
            break
    return " ".join(chosen) if chosen else "I could not find this information in the scholarship terms and conditions."


SYSTEM_PROMPT = (
    "You are a friendly RMIT scholarships advisor talking directly to a student. Answer the way a "
    "helpful staff member would speak out loud: warm, direct, plain sentences. "
    "Never say 'source', 'document', 'the text', or use citation markers like [Source 1] - the student "
    "does not see your sources, only your words. "
    "Blend the relevant facts into a short, natural answer, 2 to 5 sentences, using a short list only if "
    "the answer is genuinely a set of distinct steps or conditions. "
    "Only state facts that are actually in the material below and that actually answer the question - "
    "leave out anything unrelated, even if it appears in the material. "
    "Never invent or guess a specific number, percentage, date, or amount (like a GPA, credit "
    "point count, or dollar value) - only state one if that exact figure is written in the "
    "material; if the question asks for a figure that isn't there, say plainly that no specific "
    "figure is given rather than estimating one. "
    "The material is often a flat list of separate bullet points, each its own independent "
    "condition. Never combine two separate bullets into a new cause-and-effect or exception that "
    "isn't stated together in the same bullet - e.g. if one bullet says an action ends the "
    "scholarship and a different, unrelated bullet mentions a report, do not conclude the report "
    "is a way to avoid that outcome unless the material actually connects them. "
    "If different scholarships have different rules, give the general rule and add one short sentence "
    "noting it can vary by scholarship, rather than listing each one separately. "
    "If the material does not answer the question, reply with exactly: "
    "'I could not find this information in the scholarship terms and conditions.'"
)


def distill_concept_context(source_results: list[dict]) -> str | None:
    """Pull just the matching clause out of each chunk instead of the whole per-scholarship
    text. A small model given several full named-scholarship excerpts tends to contradict
    itself; a couple of short, deduplicated clauses let it answer plainly and consistently."""
    seen: set[str] = set()
    clauses = []
    for result in source_results:
        text = result["chunk"].text.replace("\n", " ")
        match = re.search(r"[^•]*\bany other scholarship\b[^•]*", text, re.I)
        if not match:
            continue
        clause = re.sub(r"\s+", " ", match.group(0)).strip(" •.")
        key = clause.lower()
        if key not in seen:
            seen.add(key)
            clauses.append(clause)
        if len(clauses) == 2:
            break
    if not clauses:
        return None
    bullets = "\n".join(f"- {clause}" for clause in clauses)
    return (
        "This condition is stated separately, in essentially the same words, under most "
        f"individual scholarships in the terms and conditions:\n{bullets}"
    )


def ollama_answer(question: str, results: list[dict], model: str, base_url: str, context: str | None = None) -> str | None:
    if context is None:
        context = "\n\n".join(f"{r['chunk'].text}" for r in results[:4])
    prompt = f"Question: {question}\n\nRelevant material:\n{context}\n\nAnswer:"
    payload = json.dumps(
        {
            "model": model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }
    ).encode()
    request = urllib.request.Request(base_url.rstrip("/") + "/api/generate", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read())["response"].strip()
    except (OSError, KeyError, ValueError, urllib.error.URLError):
        return None


ABOUT_ANSWER = (
    "I answer questions about RMIT coursework scholarships and grants, based on the August 2026 "
    "terms and conditions. Ask me things like whether you can study part-time, what happens if you "
    "take leave, how payments and tax work, GPA requirements, or the conditions for a specific "
    "named scholarship."
)
META_QUESTION_RE = re.compile(
    r"^(what('?s| is) (this( chat| bot| assistant)?|walert)( (about|for))?\??|"
    r"who are you\??|"
    r"what (can|do) you (do|help( with)?|answer)\??|"
    r"(hi|hello|hey)[!.]?)$"
)


def is_meta_question(question: str) -> bool:
    normalized = re.sub(r"\s+", " ", question.strip().lower()).rstrip("?.! ")
    return bool(META_QUESTION_RE.match(normalized))


class ScholarshipRAG:
    def __init__(self, chunks: list[Chunk]):
        self.retriever = HybridRetriever(chunks)

    def ask(self, question: str, method: str = "hybrid", use_ollama: bool = True) -> dict:
        if is_meta_question(question):
            return {"answer": ABOUT_ANSWER, "answerable": True, "generator": "none", "sources": []}
        results = self.retriever.search(question, top_k=5, method=method)
        # Require meaningful lexical evidence, not merely a relative rank.
        answerable = is_answerable(results)
        if not answerable:
            return {"answer": "I could not find this information in the scholarship terms and conditions.", "answerable": False,
                    "generator": "none", "sources": []}
        concept_context = None
        if results[0].get("concept_match"):
            # This fact is repeated near-identically across many per-scholarship sections rather
            # than stated once, so several matches corroborating the same wording is normal, not a
            # sign of unrelated content bleeding in the way a mismatched section would be.
            source_results = [r for r in results if r.get("concept_match")][:3]
            concept_context = distill_concept_context(source_results)
        else:
            # Keep the LLM's evidence identical to what's shown as citations, so it can't blend in
            # a number or condition from an unrelated, lower-ranked scholarship section.
            source_results = [r for r in results if r["chunk"].section == results[0]["chunk"].section][:3]
        answer = None
        generator = "extractive"
        if use_ollama:
            answer = ollama_answer(
                question, source_results, os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
                os.getenv("OLLAMA_URL", "http://localhost:11434"), context=concept_context,
            )
            if answer:
                generator = "ollama"
        answer = answer or extractive_answer(question, results)
        sources = [
            {"id": r["chunk"].id, "page": r["chunk"].page, "section": r["chunk"].section,
             "title": r["chunk"].title, "citation": r["chunk"].citation,
             "excerpt": r["chunk"].text[:700], "score": round(r["confidence"], 4)}
            for r in source_results
        ]
        return {"answer": answer, "answerable": True, "generator": generator, "sources": sources}


def is_answerable(results: list[dict]) -> bool:
    """Conservative calibrated gate; tune only against a held-out validation set."""
    return bool(results and (results[0].get("route_match") or results[0].get("concept_match") or results[0]["bm25_score"] >= 9.0))
