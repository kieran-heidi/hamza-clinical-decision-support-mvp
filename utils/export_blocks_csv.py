#!/usr/bin/env python3
import csv, json, argparse
from pathlib import Path

FIELDS = ["topic_num","topic_title","heading","subheading","grade",
          "section_num","subsection_num","page_start","page_end","text"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="blocks json")
    ap.add_argument("dst", help="csv path (e.g., data/blocks_review.csv)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = json.loads(Path(args.src).read_text())
    if args.limit: rows = rows[:args.limit]
    with open(args.dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"Wrote {len(rows)} rows → {args.dst}")
