"""Build the hybrid BM25 + FAISS index over MSMARCO-XI sample records.

Run with:  python -m scripts.build_index [--data-dir DIR] [--index-dir DIR]

Reads every ``*.jsonl`` in the data dir, runs the chunker with all four
strategies, embeds with the hashing embedder, builds BM25, and saves
the index files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make ``app`` importable when this is run as ``python -m scripts.build_index``
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR, INDEX_DIR  # noqa: E402
from app.retriever import HybridRetriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--index-dir", default=str(INDEX_DIR))
    ap.add_argument("--limit", type=int, default=0, help="limit records per file (0 = all)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        print(f"no .jsonl files in {data_dir}", file=sys.stderr)
        sys.exit(1)

    records = []
    t0 = time.time()
    for f in files:
        with f.open("r", encoding="utf-8") as fp:
            file_records = []
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                file_records.append(json.loads(line))
                if args.limit and len(file_records) >= args.limit:
                    break
        print(f"  {f.name}: {len(file_records)} records", flush=True)
        records.extend(file_records)
    print(f"Total: {len(records)} records in {time.time()-t0:.1f}s", flush=True)

    HybridRetriever.build_from_records(records, index_dir=index_dir)

    # quick sanity check
    r = HybridRetriever.load(index_dir=index_dir)
    print(f"Index ready: {len(r.chunks)} chunks")
    sample_q = records[0].get("query") or records[0].get("Eng_Query") or "what is a corporation"
    hits = r.retrieve(sample_q, top_k=3)
    print(f"Top-3 for sample query '{sample_q[:60]}':")
    for ch, score, meta in hits:
        print(f"  [{score:.3f}] {ch.short(100)}")


if __name__ == "__main__":
    main()
