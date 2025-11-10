#!/usr/bin/env python3
"""
Parse a chapter PDF into structured blocks using a config file.

Usage:
  python scripts/pdf_parser.py \
    --pdf data/respiratory.pdf \
    --config config/pdf_parser.yaml \
    --out data/blocks.json
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse, json, re, yaml
from pypdf import PdfReader

# --- Globals filled by config ---
HEADER_LINES: set[str] = set()
RE_PAGE_LABEL: Optional[re.Pattern] = None
NUMERIC_TOPICS: Dict[str, str] = {}
LOGICAL_HEADINGS: set[str] = set()
INNER_SUBS: Dict[str, set[str]] = {}

# --- Regex helpers ---
RE_NUMERIC_START = re.compile(r"^(\d+(?:\.\d+)+)\b")  # captures "17", "17.1", "17.3.4.2.2"
RE_ICD_ONLY = re.compile(r"^[A-Z]\d[\w\.\-/]*(?:/[A-Z]?\d[\w\.\-/]*)+$")  # J18.0-2/J18.8-9
RE_GRADE = re.compile(r"^Grade\s+([1-9][0-9]*)$", re.IGNORECASE)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def load_config(cfg_path: Path) -> None:
    global HEADER_LINES, RE_PAGE_LABEL, NUMERIC_TOPICS, LOGICAL_HEADINGS, INNER_SUBS
    cfg = yaml.safe_load(cfg_path.read_text())
    # page furniture
    hdrs = cfg.get("page_headers", [])
    ch = cfg.get("chapter")
    if ch:
        hdrs.append(f"CHAPTER {ch}")
    HEADER_LINES = {h.upper() for h in hdrs}
    r = cfg.get("strip_page_labels_regex")
    RE_PAGE_LABEL = re.compile(r) if r else None
    # topics + headings
    NUMERIC_TOPICS = {k: _norm(v).upper() for k, v in cfg["numeric_topics"].items()}
    LOGICAL_HEADINGS = {_norm(h).upper() for h in cfg.get("logical_headings", [])}
    INNER_SUBS = {}
    for k, vals in (cfg.get("inner_subheadings") or {}).items():
        INNER_SUBS[_norm(k).upper()] = {_norm(v).upper() for v in (vals or [])}

def normalize_lines(raw: str) -> List[str]:
    raw = raw.replace("\r\n", "\n")
    raw = re.sub(r"[ \t]+\n", "\n", raw)         # trim trailing blanks
    raw = re.sub(r"(\S)-\n(\S)", r"\1\2", raw)   # de-hyphenate
    raw = re.sub(r"\n{3,}", "\n\n", raw)         # collapse large gaps
    lines = [ln.rstrip() for ln in raw.split("\n")]

    out: List[str] = []
    for ln in lines:
        t = ln.strip()
        up = t.upper()
        if up in HEADER_LINES:
            continue
        if RE_PAGE_LABEL and RE_PAGE_LABEL.match(t):
            continue
        out.append(ln)
    return out

def coalesce_topic_title(lines: List[str], j: int) -> Tuple[str, int, Optional[str]]:
    """
    Starting at lines[j] which begins with a numeric topic key (e.g., 17.1.1),
    extend title with an optional short uppercase continuation line,
    and capture a following ICD-only line as icd_codes.
    Returns: (title_text_after_number, next_index, icd_codes or None)
    """
    # Strip the numeric prefix; keep remainder as a tentative title
    line = lines[j].strip()
    m = RE_NUMERIC_START.match(line)
    title_after = line[m.end():].strip() if m else line
    i = j + 1
    icd = None

    # Short UPPERCASE continuation (e.g., "ADULTS") on next line?
    if i < len(lines):
        nxt = lines[i].strip()
        if nxt and nxt.upper() == nxt and len(nxt.split()) <= 8 \
           and not RE_NUMERIC_START.match(nxt) and not RE_ICD_ONLY.match(nxt):
            title_after = f"{title_after} {nxt}".strip()
            i += 1

    # ICD-only on next line?
    if i < len(lines):
        nxt = lines[i].strip()
        if RE_ICD_ONLY.match(nxt):
            icd = nxt
            i += 1

    return title_after, i, icd

def parse(pdf_path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    blocks: List[Dict[str, Any]] = []

    path_nums: List[str] = []      # ["17","17.1","17.1.1", ...]
    path_titles: List[str] = []    # matching configured titles
    buf: List[str] = []
    buf_start: Optional[int] = None
    heading: Optional[str] = None
    subheading: Optional[str] = None
    grade: Optional[str] = None
    current_icd: Optional[str] = None

    def start(pi: int):
        nonlocal buf, buf_start
        buf, buf_start = [], pi

    def flush(pi: int):
        nonlocal buf, buf_start, heading, subheading, grade, current_icd
        if not buf:
            return
        text = "\n".join(buf).strip()
        if not text:
            buf, buf_start = [], None
            return
        md: Dict[str, Any] = {
            "text": text,
            "page_start": buf_start or pi,
            "page_end": pi,
            "path_nums": path_nums[:],
            "path_titles": path_titles[:],
            "topic_num": path_nums[-1] if path_nums else None,
            "topic_title": path_titles[-1] if path_titles else None,
            "heading": heading,
            "subheading": subheading,
            "grade": grade,
            "icd_codes": current_icd,
            # convenience filters
            "section_num": path_nums[1] if len(path_nums) > 1 else (path_nums[0] if path_nums else None),
            "subsection_num": path_nums[2] if len(path_nums) > 2 else None,
        }
        blocks.append(md)
        buf, buf_start = [], None

    for pi, page in enumerate(reader.pages, start=1):
        lines = normalize_lines(page.extract_text() or "")
        j = 0
        while j < len(lines):
            line = lines[j].strip()
            if not line:
                if buf_start is None: start(pi)
                buf.append(line); j += 1; continue

            # Numeric topic (only if it's explicitly listed in config)
            m = RE_NUMERIC_START.match(line)
            if m and m.group(1) in NUMERIC_TOPICS:
                flush(pi)
                num = m.group(1)
                title_after, next_j, icd = coalesce_topic_title(lines, j)
                # Set hierarchy to this exact node
                segs = num.split(".")
                path_nums = [".".join(segs[:i+1]) for i in range(len(segs))]
                # Titles follow config (parents remain; replace current)
                path_titles = path_titles[:len(segs)-1] + [NUMERIC_TOPICS[num]]
                # Reset section context
                heading = None; subheading = None; grade = None
                current_icd = icd
                start(pi)
                j = next_j
                continue

            # Logical headings (config-driven)
            up = _norm(line).upper()
            if up in LOGICAL_HEADINGS and len(up.split()) <= 6:
                flush(pi)
                heading = up
                subheading = None
                grade = None
                start(pi)
                j += 1
                continue

            # Grade N
            mg = RE_GRADE.match(line)
            if mg:
                flush(pi)
                heading = "GRADE"; subheading = None; grade = mg.group(1)
                start(pi)
                j += 1
                continue

            # Inner subheadings (only inside a logical block)
            if heading:
                allowed = set()
                allowed |= INNER_SUBS.get(heading, set())
                # Allow inner subheads keyed by topic title (if you choose to use it)
                if path_titles:
                    allowed |= INNER_SUBS.get(path_titles[-1].upper(), set())
                if up in allowed:
                    flush(pi)
                    subheading = line.strip()
                    start(pi)
                    j += 1
                    continue

            # Content
            if buf_start is None: start(pi)
            buf.append(line)
            j += 1

    flush(len(reader.pages))
    return blocks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to chapter PDF (e.g., data/respiratory.pdf)")
    ap.add_argument("--config", default="config/pdf_parser.yaml", help="YAML config path")
    ap.add_argument("--out", default="data/blocks.json", help="Output JSON path")
    args = ap.parse_args()

    load_config(Path(args.config))
    out = parse(Path(args.pdf))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Wrote {len(out)} blocks → {args.out}")

if __name__ == "__main__":
    main()
