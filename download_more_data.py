"""Stream a single language's MSMARCO-XI parquet into a JSONL sample.

This uses pyarrow's row-by-row slicing to avoid loading the full 461 MB
file into memory, which would otherwise OOM on small sandboxes (1-2 GB).

Usage:
    python -m scripts.download_more_data --lang hi --n 5000 --out data/hindi_5k.jsonl
    python -m scripts.download_more_data --lang mr --n 1000 --out data/marathi_1k.jsonl

If you also want the per-language translated passages (for stronger
multilingual BM25), pass --include-translations.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

# language code -> filename mapping
LANG_FILES = {
    "hi": "hinval", "bn": "benval", "gu": "gujval", "kn": "kanval",
    "ml": "malval", "mr": "marval", "ne": "nepval", "or": "orival",
    "pa": "panval", "sa": "sanval", "ta": "tamval", "te": "telval",
    "ur": "urdval", "as": "asmval",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(LANG_FILES))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-translations", action="store_true",
                    help="include passages.Translated_passages (uses more memory)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "query_id", "source_lang", "target_lang", "query_type",
        "query", "Answer", "Eng_Query", "Eng_Answer",
        "passages.is_selected", "passages.English_passages",
    ]
    if args.include_translations:
        columns.append("passages.Translated_passages")

    print(f"Downloading validation/{LANG_FILES[args.lang]}.parquet ...", flush=True)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        filename=f"validation/{LANG_FILES[args.lang]}.parquet",
        repo_type="dataset",
        local_dir=str(out.parent),
    )
    print(f"  {os.path.getsize(path)//1024//1024}MB", flush=True)

    import pyarrow.parquet as pq
    import resource
    pf = pq.ParquetFile(path)
    n_total = pf.metadata.num_rows
    print(f"  {n_total} rows in source", flush=True)

    if args.include_translations:
        # With all columns, even row-group read can OOM on 1-2GB sandboxes.
        # Process in small row slices that we keep only one at a time.
        random.seed(args.seed)
        want = sorted(random.sample(range(n_total), min(args.n, n_total)))
        keep = set(want)
        rg = pf.read_row_group(0, columns=columns)
        print(f"  read row group: {rg.num_rows} rows, nbytes={rg.nbytes/1024/1024:.0f}MB, "
              f"rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB", flush=True)
        n_kept = 0
        t = time.time()
        with out.open("w", encoding="utf-8") as f:
            for i in range(rg.num_rows):
                if i in keep:
                    row = rg.slice(i, 1).to_pylist()[0]
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_kept += 1
                    if n_kept % 100 == 0:
                        print(f"    kept {n_kept}/{len(want)}, "
                              f"rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB", flush=True)
                    if n_kept >= len(want):
                        break
        del rg
    else:
        # Without translations, we can usually read the full row group.
        rg = pf.read_row_group(0, columns=columns)
        n = min(args.n, rg.num_rows)
        random.seed(args.seed)
        idx = sorted(random.sample(range(rg.num_rows), n))
        rows = [rg.slice(i, 1).to_pylist()[0] for i in idx]
        del rg
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_kept = len(rows)

    os.remove(path)
    print(f"\nWrote {n_kept} records to {out}", flush=True)
    if not args.include_translations:
        print("(no Translated_passages -- cross-lingual BM25 will be limited)", flush=True)


if __name__ == "__main__":
    main()
