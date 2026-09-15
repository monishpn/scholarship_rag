"""PDF ingestion plus BM25 and TF-IDF retrieval for the scholarship document."""
from __future__ import annotations
import json, math, re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?")
SECTION_RE = re.compile(r"(?m)^(\d+(?:\.\d+)*)(?:\.)?\s+([^\n]{3,120})$")
STOPWORDS = {"a","an","and","are","as","at","be","by","can","do","for","from","how","i","if","in","is","it","my","of","on","or","that","the","this","to","what","when","where","which","who","will","with","you","your","rmit"}

def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]

def clean_page_text(text: str) -> str:
    lines = []
    for line in text.replace("\u00a0", " ").splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            lines.append(""); continue
        if re.match(r"^(Document:|Author:|Version |Save Date:|Page \d+ of \d+)", line):
            continue
        lines.append(line)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", "\n".join(lines))
    return re.sub(r"\n{3,}", "\n\n", text).strip()

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
        piece = words[start:start + size]
        if piece: yield piece
        if start + size >= len(words): break

def ingest_pdf(pdf_path: Path, output_path: Path, chunk_words: int = 230, overlap: int = 45) -> list[Chunk]:
    from pypdf import PdfReader
    chunks, current_section, current_title, body_started = [], "", "General conditions", False
    for page_number, page in enumerate(PdfReader(str(pdf_path)).pages, start=1):
        text = clean_page_text(page.extract_text() or "")
        if not body_started:
            body_started = bool(re.search(r"(?m)^1\.\s+Introduction\s*$", text) and "Contents" not in text)
        if not body_started or not text: continue
        matches = [m for m in SECTION_RE.finditer(text) if len(re.findall(r"\d+(?:\.\d+)?", m.group(2))) < 2]
        segments = []
        if matches:
            prefix = text[:matches[0].start()].strip()
            if prefix: segments.append((current_section, current_title, prefix))
            for index, match in enumerate(matches):
                current_section, current_title = match.group(1), match.group(2).strip()
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                segments.append((current_section, current_title, text[match.end():end].strip()))
        else:
            segments.append((current_section, current_title, text))
        for section, title, segment in segments:
            if not segment.split(): continue
            heading = f"Section {section}: {title}" if section else title
            for piece in _window(segment.split(), chunk_words, overlap):
                chunks.append(Chunk(f"P{page_number:03d}-C{len(chunks)+1:04d}", page_number, section, title, f"{heading}\n{' '.join(piece)}"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(c) for c in chunks], indent=2), encoding="utf-8")
    return chunks

def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**item) for item in json.loads(path.read_text(encoding="utf-8"))]

class LexicalRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.docs = [tokenize(f"{c.title} {c.title} {c.text}") for c in chunks]
        self.term_freqs = [Counter(doc) for doc in self.docs]
        self.lengths = [len(doc) for doc in self.docs]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths))
        df = Counter()
        for doc in self.docs: df.update(set(doc))
        count = max(1, len(self.docs))
        self.idf = {term: math.log(1 + (count-frequency+0.5)/(frequency+0.5)) for term, frequency in df.items()}

    def bm25_scores(self, query: list[str]) -> list[float]:
        k1, b, scores = 1.5, 0.75, []
        for frequencies, length in zip(self.term_freqs, self.lengths):
            score = 0.0
            for term in query:
                frequency = frequencies.get(term, 0)
                if frequency:
                    denominator = frequency + k1*(1-b+b*length/max(1,self.avg_length))
                    score += self.idf.get(term,0.0)*frequency*(k1+1)/denominator
            scores.append(score)
        return scores

    def tfidf_scores(self, query: list[str]) -> list[float]:
        qf = Counter(query)
        qvec = {term: frequency*self.idf.get(term,0.0) for term,frequency in qf.items()}
        qnorm = math.sqrt(sum(v*v for v in qvec.values())) or 1.0
        scores = []
        for frequencies in self.term_freqs:
            dot = sum(qvec.get(term,0.0)*frequency*self.idf.get(term,0.0) for term,frequency in frequencies.items())
            dnorm = math.sqrt(sum((frequency*self.idf.get(term,0.0))**2 for term,frequency in frequencies.items())) or 1.0
            scores.append(dot/(qnorm*dnorm))
        return scores

    def compare(self, question: str, top_k: int = 3) -> dict:
        query = tokenize(question)
        return {"question":question,"tokens":query,"bm25":self._ranked(self.bm25_scores(query),top_k),"tfidf":self._ranked(self.tfidf_scores(query),top_k)}

    def search(self, question: str, top_k: int = 5, method: str = "bm25") -> list[dict]:
        query = tokenize(question)
        scores = self.bm25_scores(query) if method == "bm25" else self.tfidf_scores(query)
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:top_k]
        return [{"chunk":self.chunks[i],"score":scores[i]} for i in order]

    def _ranked(self, scores: list[float], top_k: int) -> list[dict]:
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:top_k]
        return [{"rank":rank,"score":round(scores[i],6),"chunk_id":self.chunks[i].id,"page":self.chunks[i].page,"section":self.chunks[i].section,"title":self.chunks[i].title,"citation":self.chunks[i].citation} for rank,i in enumerate(order,start=1)]
