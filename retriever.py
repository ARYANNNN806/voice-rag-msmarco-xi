"""BM25 + dense hybrid retriever with reciprocal rank fusion.

Index layout (files in ``INDEX_DIR``):

* ``chunks.jsonl``    -- one Chunk per line, source of truth
* ``bm25.pkl``        -- pickled (tokenized_corpus, idf, doc_lens, avgdl)
* ``dense.npy``       -- (N, dim) float32 L2-normalized dense vectors
* ``dense.index``     -- FAISS IndexFlatIP for cosine via inner product
* ``meta.json``       -- build metadata (counts, langs, build time, ...)

Reciprocal Rank Fusion (RRF) is the simple, well-studied way to merge two
ranked lists without having to calibrate their raw scores.
"""
from __future__ import annotations

import json
import math
import pickle
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .chunking import Chunk, chunk_record
from .config import CFG, INDEX_DIR
from .embeddings import HashingEmbedder

# ----------------------------- BM25 -----------------------------
class BM25:
    """BM25Okapi with an inverted index for sub-millisecond retrieval at 150k docs.

    The naive "score every doc" implementation runs at ~1 ms per doc in
    Python, so 150k docs ≈ 1 s / query.  Using an inverted index brings
    this to O(|query| * avg_postings) and gets us under 10 ms.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self.n_docs: int = 0
        # inverted index: term -> list[(doc_idx, term_freq)]
        self.postings: dict[str, list[tuple[int, int]]] = {}

    def fit(self, corpus: Sequence[Sequence[str]]):
        self.docs = [list(d) for d in corpus]
        self.n_docs = len(self.docs)
        self.doc_lens = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0

        # Build inverted index in a single pass
        postings: dict[str, list[tuple[int, int]]] = {}
        df: Counter[str] = Counter()
        for i, d in enumerate(self.docs):
            if not d:
                continue
            tf = Counter(d)
            for term, f in tf.items():
                postings.setdefault(term, []).append((i, f))
                df[term] += 1
        self.postings = postings
        self.idf = {
            t: math.log(1 + (self.n_docs - df_t + 0.5) / (df_t + 0.5))
            for t, df_t in df.items()
        }
        return self

    def topk(self, query: Sequence[str], k: int) -> list[tuple[int, float]]:
        if not query or not self.docs or self.avgdl == 0:
            return []
        # Aggregate scores per doc via the inverted index
        scores: dict[int, float] = {}
        for q in query:
            plist = self.postings.get(q)
            if not plist:
                continue
            idf = self.idf.get(q, 0.0)
            for doc_idx, tf in plist:
                dl = self.doc_lens[doc_idx]
                denom_norm = 1 - self.b + self.b * dl / self.avgdl
                s = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * denom_norm)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + s
        if not scores:
            return []
        # heap.nlargest is faster than full sort for small k
        import heapq
        return heapq.nlargest(k, scores.items(), key=lambda x: x[1])


# ----------------------------- retriever -----------------------------
class HybridRetriever:
    def __init__(self, index_dir: Path | None = None):
        self.index_dir = Path(index_dir or INDEX_DIR)
        self.chunks: list[Chunk] = []
        self.bm25: BM25 | None = None
        self.dense: np.ndarray | None = None      # (N, dim) L2-normalized
        self.faiss_index = None
        self.embedder = HashingEmbedder()
        self._faiss = None

    # -------- persistence --------
    def save(self):
        d = self.index_dir
        d.mkdir(parents=True, exist_ok=True)
        with (d / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(_chunk_to_dict(c), ensure_ascii=False) + "\n")
        with (d / "bm25.pkl").open("wb") as f:
            # Don't pickle `docs` -- we only need postings at query time, and
            # `docs` is a huge list of token lists (50+ MB).
            pickle.dump({
                "postings": self.bm25.postings,
                "doc_lens": self.bm25.doc_lens,
                "avgdl": self.bm25.avgdl,
                "idf": self.bm25.idf,
                "k1": self.bm25.k1,
                "b": self.bm25.b,
                "n_docs": self.bm25.n_docs,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(d / "dense.npy", self.dense if self.dense is not None else np.zeros((0, 0), dtype="float32"))
        if self.faiss_index is not None:
            import faiss
            faiss.write_index(self.faiss_index, str(d / "dense.faiss"))
        with (d / "meta.json").open("w", encoding="utf-8") as f:
            json.dump({
                "n_chunks": len(self.chunks),
                "dense_dim": int(self.dense.shape[1]) if self.dense is not None else 0,
                "strategies": sorted({c.strategy for c in self.chunks}),
                "langs": sorted({c.target_lang for c in self.chunks if c.target_lang}),
            }, f)

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "HybridRetriever":
        d = Path(index_dir or INDEX_DIR)
        r = cls(index_dir=d)
        # chunks
        chunks = []
        with (d / "chunks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunks.append(_dict_to_chunk(json.loads(line)))
        r.chunks = chunks
        # bm25
        with (d / "bm25.pkl").open("rb") as f:
            blob = pickle.load(f)
        bm = BM25(k1=blob["k1"], b=blob["b"])
        bm.postings = blob["postings"]
        bm.doc_lens = blob["doc_lens"]
        bm.avgdl = blob["avgdl"]
        bm.idf = blob["idf"]
        bm.n_docs = blob.get("n_docs", len(bm.doc_lens))
        r.bm25 = bm
        # dense
        if (d / "dense.npy").exists():
            r.dense = np.load(d / "dense.npy")
        if (d / "dense.faiss").exists():
            import faiss
            r.faiss_index = faiss.read_index(str(d / "dense.faiss"))
        return r

    # -------- build --------
    @classmethod
    def build_from_records(cls, records: list[dict], index_dir: Path | None = None,
                           strategies: Sequence[str] = ("passage", "sentence_window",
                                                       "fixed_token", "semantic"),
                           progress_every: int = 1000,
                           embed_batch: int = 2048) -> "HybridRetriever":
        """Build the index from a list of MSMARCO-XI records.

        Memory-efficient: chunks are written to ``chunks.jsonl`` as they're
        created; embeddings are accumulated in a flushable buffer.  This
        keeps the working set bounded regardless of corpus size.
        """
        from .chunking import tokenize
        import gc

        t0 = time.time()
        r = cls(index_dir=index_dir)
        d = r.index_dir
        d.mkdir(parents=True, exist_ok=True)
        # Always start with a clean index
        for fn in ("chunks.jsonl", "bm25.pkl", "dense.npy",
                   "dense.faiss", "meta.json"):
            p = d / fn
            if p.exists():
                p.unlink()

        chunks_path = d / "chunks.jsonl"
        embedder = HashingEmbedder()

        # ---- pass 1: chunk + write to disk + accumulate text for embedding ----
        all_texts: list[str] = []
        all_chunks_meta: list[Chunk] = []  # holds metadata only; text re-read from jsonl for BM25

        n_records = len(records)
        n_chunks = 0
        with chunks_path.open("w", encoding="utf-8") as cf:
            for i, rec in enumerate(records):
                for ch in chunk_record(rec):
                    cf.write(json.dumps(_chunk_to_dict(ch), ensure_ascii=False) + "\n")
                    all_texts.append(ch.text)
                    n_chunks += 1
                if progress_every and (i + 1) % progress_every == 0:
                    print(f"  chunked {i+1}/{n_records} records -> {n_chunks} chunks "
                          f"({time.time()-t0:.1f}s)", flush=True)
        print(f"  total {n_chunks} chunks from {n_records} records "
              f"({time.time()-t0:.1f}s)", flush=True)
        r.chunks = []  # populated on load

        # ---- pass 2: compute dense embeddings in batches, write to npy ----
        t1 = time.time()
        N = len(all_texts)
        dim = embedder.dim
        # write dense matrix in float16 to halve memory and disk
        out_npy = d / "dense.npy"
        mm = np.lib.format.open_memmap(out_npy, mode="w+", dtype="float16", shape=(N, dim))
        buf_texts: list[str] = []
        for i, t in enumerate(all_texts):
            buf_texts.append(t)
            if len(buf_texts) >= embed_batch or i == N - 1:
                vecs = embedder.encode_bulk_fast(buf_texts)
                # float32 -> float16
                mm[i - len(buf_texts) + 1: i + 1] = vecs.astype("float16")
                buf_texts.clear()
            if (i + 1) % 20000 == 0:
                print(f"    embedded {i+1}/{N} ({time.time()-t1:.1f}s)", flush=True)
        del mm
        del all_texts
        gc.collect()
        print(f"  dense embeddings done ({time.time()-t1:.1f}s)", flush=True)

        # ---- pass 3: read chunks back, build BM25 ----
        t2 = time.time()
        # we need the full chunks list to populate r.chunks for retrieval
        chunks_loaded: list[Chunk] = []
        with chunks_path.open("r", encoding="utf-8") as cf:
            for line in cf:
                if line.strip():
                    chunks_loaded.append(_dict_to_chunk(json.loads(line)))
        r.chunks = chunks_loaded
        corpus = [tokenize(c.text) for c in chunks_loaded]
        r.bm25 = BM25(k1=CFG.retrieval.bm25_k1, b=CFG.retrieval.bm25_b).fit(corpus)
        # docs is no longer needed; postings carry the inverted index
        r.bm25.docs = []
        del corpus
        gc.collect()
        print(f"  BM25 fit ({time.time()-t2:.1f}s)", flush=True)

        # ---- pass 4: load dense, build FAISS (HNSW for sub-millisecond search) ----
        t3 = time.time()
        # Load dense as float16 (half memory) and add directly to FAISS in
        # float16 -- the dot-product error from fp16 is well below the
        # quantization noise of the hash embedder.
        r.dense = np.load(out_npy)
        try:
            import faiss
            # HNSW: ~0.1ms per query at 147k vectors, ~25s build.
            # Build from float16 directly to save memory; the HNSW graph
            # stores its own working-precision copy.
            index = faiss.IndexHNSWFlat(r.dense.shape[1], 32,
                                        faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 64
            index.hnsw.efSearch = 32
            # Add in chunks so we don't have to hold a separate full-size
            # float32 copy during build.
            add_bs = 8192
            for s in range(0, len(r.dense), add_bs):
                batch = r.dense[s:s + add_bs]
                # re-normalize each batch on the fly (float16 lost a little precision)
                n = np.linalg.norm(batch, axis=1, keepdims=True)
                n[n == 0] = 1.0
                batch = (batch / n).astype("float32")
                index.add(batch)
                del batch
                if (s // add_bs) % 5 == 0:
                    print(f"    hnsw add {s}/{len(r.dense)} ({time.time()-t3:.1f}s)", flush=True)
            r.faiss_index = index
            print(f"  FAISS HNSW index built ({time.time()-t3:.1f}s)", flush=True)
        except Exception as e:
            print("WARN: HNSW failed, falling back to flat:", e)
            try:
                import faiss
                index = faiss.IndexFlatIP(r.dense.shape[1])
                add_bs = 8192
                for s in range(0, len(r.dense), add_bs):
                    batch = r.dense[s:s + add_bs]
                    n = np.linalg.norm(batch, axis=1, keepdims=True)
                    n[n == 0] = 1.0
                    index.add((batch / n).astype("float32"))
                    del batch
                r.faiss_index = index
            except Exception as e2:
                print("WARN: faiss unavailable, falling back to numpy:", e2)
                r.faiss_index = None
        # We don't need r.dense at query time (FAISS has the vectors) but
        # we do want it for the save() call.  Convert to fp16 first to
        # halve memory if it isn't already.
        if r.dense is not None and r.dense.dtype != np.float16:
            r.dense = r.dense.astype("float16")

        r.embedder = embedder
        r.save()
        print(f"Built index: {len(r.chunks)} chunks in {time.time()-t0:.1f}s", flush=True)
        return r

    # -------- query --------
    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[Chunk, float, dict]]:
        """Return top-k chunks with combined score and per-channel scores."""
        cfg = CFG.retrieval
        k = top_k or cfg.top_k_final
        if not query.strip() or not self.chunks:
            return []

        # 1) BM25
        from .chunking import tokenize
        qt = tokenize(query)
        bm25_hits = self.bm25.topk(qt, cfg.top_k_bm25) if self.bm25 else []
        bm25_rank = {i: 1.0 / (cfg.rrf_k + r + 1)
                     for r, (i, _s) in enumerate(bm25_hits)}
        bm25_raw = {i: s for i, s in bm25_hits}

        # 2) Dense
        qv = self.embedder.encode_one(query)
        if self.faiss_index is not None:
            scores, idxs = self.faiss_index.search(
                qv.reshape(1, -1).astype("float32"), cfg.top_k_dense)
            dense_hits = list(zip(idxs[0].tolist(), scores[0].tolist()))
        elif self.dense is not None and len(self.dense) > 0:
            sims = (self.dense @ qv).astype("float32")
            top = np.argsort(-sims)[:cfg.top_k_dense]
            dense_hits = [(int(i), float(sims[i])) for i in top]
        else:
            dense_hits = []
        dense_rank = {i: 1.0 / (cfg.rrf_k + r + 1) for r, (i, _s) in enumerate(dense_hits)}
        dense_raw = {i: s for i, s in dense_hits}

        # 3) Reciprocal rank fusion + linear blend
        all_idx = set(bm25_rank) | set(dense_rank)
        blended = []
        for i in all_idx:
            score = (cfg.bm25_weight * bm25_rank.get(i, 0.0)
                     + cfg.dense_weight * dense_rank.get(i, 0.0))
            blended.append((i, score, bm25_raw.get(i, 0.0), dense_raw.get(i, 0.0)))
        blended.sort(key=lambda x: x[1], reverse=True)

        out = []
        for i, score, bm_raw, de_raw in blended[:k]:
            ch = self.chunks[i]
            out.append((ch, float(score), {
                "bm25_raw": float(bm_raw),
                "dense_raw": float(de_raw),
                "rrf_bm25": float(bm25_rank.get(i, 0.0)),
                "rrf_dense": float(dense_rank.get(i, 0.0)),
                "idx": i,
            }))
        return out


# ----------------------------- helpers -----------------------------
def _chunk_to_dict(c: Chunk) -> dict:
    return asdict(c)


def _dict_to_chunk(d: dict) -> Chunk:
    return Chunk(**d)
