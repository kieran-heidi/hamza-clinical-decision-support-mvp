#!/usr/bin/env python3
# usage:
#   python scripts/split_pdf.py data/full-stg-guidelines-2024.pdf 120-160 data/respiratory.pdf
# or multiple ranges:
#   python scripts/split_pdf.py data/full-stg-guidelines-2024.pdf 120-160,165-200 data/respiratory.pdf
import sys, fitz

def parse_ranges(r):
    # e.g. "120-160,165-200"
    out = []
    for part in r.split(","):
        a,b = part.split("-")
        out.append((int(a), int(b)))
    return out

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: split_pdf.py INPUT.pdf RANGES OUTPUT.pdf")
        sys.exit(1)
    src, ranges, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    doc = fitz.open(src)
    out = fitz.open()
    for a,b in parse_ranges(ranges):
        # fitz uses 0-based indices; user supplies 1-based
        for p in range(a-1, b):
            out.insert_pdf(doc, from_page=p, to_page=p)
    out.save(dst)
    out.close(); doc.close()
    print(f"Wrote {dst}")
