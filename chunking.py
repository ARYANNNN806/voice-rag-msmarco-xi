"""Chunking strategies for MSMARCO-XI passages.

The brief explicitly asks for "vast" chunking -- not a single naive approach.
We expose four complementary strategies and a `Chunker` orchestrator that
combines them at index time so that retrieval can pick the best match.

Why four? Each strategy has a different failure mode:

* ``passage``         -- one chunk per dataset passage.  Coarse but matches
                         how MSMARCO was judged (the `is_selected` flag is per
                         passage), so we always index it.
* ``sentence_window`` -- each sentence is its own chunk with k-sentence
                         overlap. Best for factoid QA.
* ``fixed_token``     -- 80-token windows with 20-token overlap. Fallback for
                         long passages / boilerplate.
* ``semantic``        -- splits on sentence boundaries when the embedding
                         centroid shift between consecutive sentences exceeds
                         a threshold. Adaptive to content density.

Every chunk carries metadata so retrieval can filter and so the harness can
show provenance in answers.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

# Use the `regex` module for variable-width look-behinds; fall back to stdlib.
try:
    import regex as _re
    _HAS_REGEX = True
except Exception:
    _re = re
    _HAS_REGEX = False

# --- sentence boundary heuristic ---
# Handles Latin, Devanagari (hi, mr, ne, sa), Bengali (bn), Gurmukhi (pa),
# Gujarati (gu), Oriya (or), Tamil (ta), Telugu (te), Kannada (kn),
# Malayalam (ml), Assamese (as), and Urdu.
# Purna virama (।) U+0964 is the Indic full stop; arabic full stop (۔) U+06D4
# is also handled.  We use the `regex` module because the stdlib re doesn't
# support variable-width look-behinds (some scripts have multi-codepoint
# terminators).
_SENT_SPLIT = _re.compile(
    r"(?<=[.!?۔])\s+|"
    r"(?<=[।])\s*|"
    r"(?<=[\n]{2,})"
)


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    text = text.replace("\r", " ")
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# --- token-ish approximation ---
# Real Indic tokenizers would help but a unicode-aware word split is fine for
# chunking (we're not training anything here).
_TOKEN_RE = re.compile(r"[\w\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0600-\u06FF]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def detok_len(text: str) -> int:
    """Approximate token count (cheap; good enough for fixed-size windows)."""
    return len(tokenize(text))


# ---------- chunk dataclass ----------
@dataclass
class Chunk:
    chunk_id: str
    text: str
    strategy: str
    source_passage_idx: int
    source_query_id: int
    source_query_en: str
    source_query_lang: str
    target_lang: str
    is_gold: bool                       # matches dataset's is_selected flag
    n_tokens: int
    n_chars: int
    meta: dict = field(default_factory=dict)

    def short(self, n: int = 120) -> str:
        s = self.text.replace("\n", " ")
        return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "..."


# ---------- individual strategies ----------
def chunk_passage(text: str, **ctx) -> List[Chunk]:
    """One chunk per passage.  Coarsest, but matches MSMARCO's gold labels."""
    if not text or not text.strip():
        return []
    return [_mk_chunk(text, "passage", 0, ctx)]


def chunk_sentence_window(text: str, window: int = 3, stride: int = 1, **ctx) -> List[Chunk]:
    """Sliding window of N consecutive sentences.  Overlap = window - stride."""
    sents = split_sentences(text)
    if not sents:
        return []
    if len(sents) <= window:
        return [_mk_chunk(" ".join(sents), "sentence_window", 0, ctx)]
    out: List[Chunk] = []
    for start in range(0, len(sents) - window + 1, stride):
        piece = " ".join(sents[start:start + window])
        out.append(_mk_chunk(piece, "sentence_window", start, ctx))
    return out


def chunk_fixed_token(text: str, size: int = 80, overlap: int = 20, **ctx) -> List[Chunk]:
    """Fixed token-size windows with overlap.  Robust to long boilerplate."""
    toks = tokenize(text)
    if not toks:
        return []
    if len(toks) <= size:
        return [_mk_chunk(" ".join(toks), "fixed_token", 0, ctx)]
    step = size - overlap
    out: List[Chunk] = []
    for start in range(0, len(toks), step):
        piece_toks = toks[start:start + size]
        out.append(_mk_chunk(" ".join(piece_toks), "fixed_token", start, ctx))
        if start + size >= len(toks):
            break
    return out


def chunk_semantic(text: str, sim_threshold: float = 0.15, **ctx) -> List[Chunk]:
    """Split at sentence boundaries where the embedding shift crosses a threshold.

    Uses the same hashing embedder as the retriever for consistency, so a
    chunk boundary is by construction a place where the topic shifts in the
    same space retrieval happens in.
    """
    sents = split_sentences(text)
    if len(sents) <= 2:
        return chunk_passage(text, **ctx)

    # Local import to avoid a cycle at module load
    from .embeddings import HashingEmbedder
    embedder = HashingEmbedder()
    vecs = embedder.encode(sents)

    # cosine-similarity between consecutive sentences
    def _cos(a, b):
        import numpy as np
        na = np.linalg.norm(a) or 1.0
        nb = np.linalg.norm(b) or 1.0
        return float(a @ b / (na * nb))

    boundaries = [0]
    for i in range(1, len(sents)):
        sim = _cos(vecs[i - 1], vecs[i])
        if sim < sim_threshold:
            boundaries.append(i)
    boundaries.append(len(sents))

    out: List[Chunk] = []
    for j in range(len(boundaries) - 1):
        a, b = boundaries[j], boundaries[j + 1]
        piece = " ".join(sents[a:b])
        if piece.strip():
            out.append(_mk_chunk(piece, "semantic", a, ctx))
    return out


# ---------- helpers ----------
def _mk_chunk(text: str, strategy: str, passage_idx: int, ctx: dict) -> Chunk:
    cid_src = f"{ctx.get('query_id','?')}::{passage_idx}::{strategy}::{text[:64]}"
    cid = hashlib.md5(cid_src.encode("utf-8")).hexdigest()[:16]
    return Chunk(
        chunk_id=cid,
        text=text,
        strategy=strategy,
        source_passage_idx=passage_idx,
        source_query_id=ctx.get("query_id", 0),
        source_query_en=ctx.get("eng_query", ""),
        source_query_lang=ctx.get("query_lang", ""),
        target_lang=ctx.get("target_lang", ""),
        is_gold=ctx.get("is_gold", False),
        n_tokens=len(tokenize(text)),
        n_chars=len(text),
    )


# ---------- orchestrator ----------
STRATEGIES = {
    "passage":         {"fn": chunk_passage, "always": True},
    "sentence_window": {"fn": lambda t, **c: chunk_sentence_window(t, window=3, stride=1, **c)},
    "fixed_token":     {"fn": lambda t, **c: chunk_fixed_token(t, size=80, overlap=20, **c)},
    "semantic":        {"fn": lambda t, **c: chunk_semantic(t, sim_threshold=0.18, **c)},
}


def chunk_text(text: str, *, query_id: int, eng_query: str, query_lang: str,
               target_lang: str, passage_idx: int, is_gold: bool,
               strategies: Sequence[str] = ("passage", "sentence_window",
                                            "fixed_token", "semantic")) -> List[Chunk]:
    """Run all requested strategies on one passage, dedup near-identical chunks."""
    ctx = {
        "query_id": query_id,
        "eng_query": eng_query,
        "query_lang": query_lang,
        "target_lang": target_lang,
        "is_gold": is_gold,
    }
    seen: dict[str, Chunk] = {}
    for s in strategies:
        for ch in STRATEGIES[s]["fn"](text, passage_idx=passage_idx, **ctx):
            if ch.chunk_id not in seen:
                seen[ch.chunk_id] = ch
    return list(seen.values())


def chunk_record(rec: dict, include_translations: bool = True) -> List[Chunk]:
    """Chunk one MSMARCO-XI record across all its passages.

    When ``include_translations`` is True (default), the translated passages
    in the record's own language are also chunked.  This is what makes the
    index truly multilingual: for a Hindi query, BM25 can match against
    the Hindi translated passage directly; for an English query, the
    English passages dominate.  The English passage is always included as
    a "cross-lingual" anchor.
    """
    passages = rec.get("passages", {})
    eng_passages = passages.get("English_passages", []) or []
    sel = passages.get("is_selected", []) or [0] * len(eng_passages)
    if not eng_passages:
        return []

    target_lang = rec.get("target_lang", "")
    translated = passages.get("Translated_passages", []) or []
    # Translated passages are aligned by index with English passages.

    out: List[Chunk] = []
    for i, p in enumerate(eng_passages):
        if not p or not p.strip():
            continue
        out.extend(chunk_text(
            p,
            query_id=int(rec.get("query_id", 0)),
            eng_query=rec.get("Eng_Query", ""),
            query_lang=rec.get("source_lang", ""),
            target_lang="eng_Latn",
            passage_idx=i,
            is_gold=bool(sel[i] if i < len(sel) else 0),
        ))
        # Also chunk the translated version if present
        if include_translations and i < len(translated) and translated[i]:
            t = translated[i]
            if t and t.strip():
                out.extend(chunk_text(
                    t,
                    query_id=int(rec.get("query_id", 0)),
                    eng_query=rec.get("Eng_Query", ""),
                    query_lang=target_lang,
                    target_lang=target_lang,
                    passage_idx=i,
                    is_gold=bool(sel[i] if i < len(sel) else 0),
                ))
    return out
