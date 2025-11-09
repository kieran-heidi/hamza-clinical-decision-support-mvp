# ingest.py
import re, os, fitz, argparse
from dataclasses import dataclass
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------- PDF extraction & cleaning ----------
HEADER_FOOTER_RE = re.compile(
    r"(?:Page\s+\d+\s+of\s+\d+|^\s*Department of Health.*$)", re.I | re.M
)

def extract_pages(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    return [p.get_text("text") for p in doc]

def clean_page(text: str) -> str:
    t = HEADER_FOOTER_RE.sub("", text)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# ---------- Tokenization (word fallback; good enough for MVP) ----------
def tokenize_words(s: str) -> List[str]:
    return s.split()

@dataclass
class Window:
    doc_id: str
    window_id: int
    text: str
    char_start: int
    char_end: int
    page_start: int
    page_end: int

def build_windows_from_pages(doc_id: str, pages: List[str], win_words=400, overlap_words=80) -> List[Window]:
    cleaned = [clean_page(p) for p in pages]
    full = "\n\n".join(cleaned)

    # page char spans
    page_spans = []
    pos = 0
    for i, p in enumerate(cleaned):
        start = pos
        pos += len(p) + 2  # account for the \n\n join
        page_spans.append((i, start, pos))

    words = tokenize_words(full)
    step = max(1, win_words - overlap_words)
    windows: List[Window] = []
    # precompute char offsets approx by rebuilding as we slide
    # (cheap, deterministic)
    # map word index -> char offset
    chars = []
    running = 0
    for w in words:
        chars.append(running)
        running += len(w) + 1
    chars.append(running)

    w_id = 0
    for start in range(0, len(words), step):
        end = min(len(words), start + win_words)
        if start >= end:
            break
        text = " ".join(words[start:end])
        char_start = chars[start]
        char_end = chars[end]

        # pages covered
        pages_cov = [pg for pg, s, e in page_spans if not (char_end <= s or char_start >= e)]
        page_start = pages_cov[0] if pages_cov else 0
        page_end = pages_cov[-1] if pages_cov else page_start

        windows.append(Window(
            doc_id=doc_id,
            window_id=w_id,
            text=text,
            char_start=char_start,
            char_end=char_end,
            page_start=page_start,
            page_end=page_end,
        ))
        w_id += 1
        if end == len(words):
            break
    return windows

# ---------- Index to Chroma ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to guideline PDF")
    ap.add_argument("--doc-id", default="SA_STG_Paeds_2023")
    ap.add_argument("--db", default="./chroma_db")
    ap.add_argument("--win", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=80)
    args = ap.parse_args()

    pages = extract_pages(args.pdf)
    windows = build_windows_from_pages(args.doc_id, pages, args.win, args.overlap)

    client = chromadb.PersistentClient(path=args.db, settings=Settings(allow_reset=True))
    coll = client.get_or_create_collection(name="guidelines")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    ids, docs, metas, embeds = [], [], [], []
    for w in windows:
        ids.append(f"{w.doc_id}::{w.window_id}")
        docs.append(w.text)
        metas.append({
            "doc_id": w.doc_id,
            "window_id": w.window_id,
            "page_start": int(w.page_start),
            "page_end": int(w.page_end),
            "char_start": int(w.char_start),
            "char_end": int(w.char_end),
        })

    embeds = model.encode(docs, batch_size=64, show_progress_bar=True).tolist()
    # reset collection to avoid dupes on re-ingest (optional)
    coll.delete(where={"doc_id": args.doc_id})
    coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)

    print(f"Ingested {len(windows)} windows into {args.db}/guidelines")

if __name__ == "__main__":
    main()
