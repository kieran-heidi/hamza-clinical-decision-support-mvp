#!/usr/bin/env python3
import re, json, argparse
from pathlib import Path

def match(b, args):
    if args.topic and not (b.get("topic_num","").startswith(args.topic) or args.topic.lower() in (b.get("topic_title","") or "").lower()):
        return False
    if args.heading and (b.get("heading","").upper() != args.heading.upper()):
        return False
    if args.subheading and args.subheading.lower() not in (b.get("subheading","") or "").lower():
        return False
    if args.contains and not re.search(args.contains, b.get("text",""), re.IGNORECASE):
        return False
    return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--topic", help='e.g., "17.1.2" or text like "croup"')
    ap.add_argument("--heading", help='e.g., "MEDICINE TREATMENT"')
    ap.add_argument("--subheading", help='e.g., "Severe attacks"')
    ap.add_argument("--contains", help='regex to search in text')
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    blocks = json.loads(Path(args.src).read_text())
    hits = [b for b in blocks if match(b, args)]
    print(f"Found {len(hits)} blocks")
    for i, b in enumerate(hits[:args.limit], 1):
        lbl = " / ".join(x for x in [
            b.get("topic_num"), b.get("topic_title"),
            b.get("heading"), b.get("subheading")
        ] if x)
        print(f"[{i}] {lbl}  pp.{b['page_start']}-{b['page_end']}  len={len(b['text'])}")
        print("    " + b["text"][:300].replace("\n"," ") + ("…" if len(b["text"])>300 else ""))
