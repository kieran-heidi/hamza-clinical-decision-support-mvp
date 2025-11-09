# retriever.py
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def stitch_neighbors(
    hits: List[Dict[str, Any]],
    fetch_doc,
    left=1,
    right=1,
    token_cap_words=1500
) -> List[Dict[str, Any]]:
    """Stitch ± neighbors within the same doc; cap total words."""
    stitched = []
    seen = set()

    for h in hits:
        m = h["metadata"]
        key = (m["doc_id"], m["window_id"])
        if key in seen:
            continue
        seen.add(key)

        doc_id = m["doc_id"]
        center = int(m["window_id"])

        # grow window
        L, R = center, center
        texts, total_words = [], 0

        def add_window(w_id):
            nonlocal texts, total_words
            doc, meta = fetch_doc(doc_id, w_id)
            texts.append(doc)
            total_words += len(doc.split())

        # start with center
        add_window(center)

        # expand alternately left/right until cap or edges
        j = 1
        while total_words < token_cap_words and (j <= left or j <= right):
            grew = False
            if j <= left and center - j >= 0:
                add_window(center - j)
                grew = True
            if total_words >= token_cap_words:
                break
            if j <= right:
                doc, meta = fetch_doc(doc_id, center + j)
                if doc is not None:
                    add_window(center + j)
                    grew = True
            if not grew:
                break
            j += 1

        # figure page range from first/last meta we actually used
        # (simple approach: re-fetch first/last)
        first = fetch_doc(doc_id, max(center - left, 0))[1]
        last = fetch_doc(doc_id, center + right)[1] or m

        stitched.append({
            "doc_id": doc_id,
            "text": "\n".join(texts),
            "pages": (int(first.get("page_start", m["page_start"])),
                      int(last.get("page_end", m["page_end"]))),
            "base_score": h["distance"] if "distance" in h else h.get("score", 0.0),
        })
    return stitched

class GuidelineRetriever:
    def __init__(self, db_path="./chroma_db", collection="guidelines"):
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings())
        self.coll = self.client.get_or_create_collection(name=collection)
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _fetch_by_id(self, id_str) -> Tuple[str, Dict[str, Any]]:
        res = self.coll.get(ids=[id_str])
        if not res["ids"]:
            return None, None
        return res["documents"][0], res["metadatas"][0]

    def fetch_doc(self, doc_id: str, window_id: int):
        id_str = f"{doc_id}::{window_id}"
        return self._fetch_by_id(id_str)

    def query(self, q: str, k=4, doc_id_filter=None):
        q_embed = self.embedder.encode([q]).tolist()
        where = {"doc_id": doc_id_filter} if doc_id_filter else {}
        res = self.coll.query(
            query_embeddings=q_embed, n_results=k, where=where, include=["documents", "metadatas", "distances"]
        )
        hits = []
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            })
        return hits

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--doc-id", default="SA_STG_Paeds_2023")
    ap.add_argument("--db", default="./chroma_db")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    r = GuidelineRetriever(db_path=args.db)
    hits = r.query(args.q, k=args.k, doc_id_filter=args.doc_id)
    def fetch(doc_id, window_id): return r.fetch_doc(doc_id, window_id)
    stitched = stitch_neighbors(hits, fetch, left=1, right=1, token_cap_words=1500)

    print("\n=== TOP HITS ===")
    for h in hits:
        m = h["metadata"]
        print(f"- {m['doc_id']}::{m['window_id']}  p.{m['page_start']}-{m['page_end']}  dist={h['distance']:.4f}")

    print("\n=== STITCHED WINDOW (first) ===")
    if stitched:
        s0 = stitched[0]
        print(f"[pages {s0['pages'][0]}–{s0['pages'][1]}]\n")
        print(s0["text"][:1200], "...\n")
