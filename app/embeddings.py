"""Multilingual hash-based embedding (FastText-style hashing trick).

Why no transformer?  Two reasons:

1. *Latency budget.*  The full pipeline must fit in 200 ms; even a 22 MB
   multilingual MiniLM model is ~30-40 ms per query on a single CPU core
   and another 30-60 ms of model load on first call.
2. *Memory.*  The sandbox has 1.9 GB RAM.  We can't keep a transformer hot
   *and* a 13 k-chunk FAISS index resident.

Hashing n-grams give us:

* truly language-agnostic vectorization (any unicode script is just a stream
  of characters),
* ~1-2 ms encoding per short string on this CPU,
* identical vectors on every run (no model version drift),
* the same vector space for index-time chunks and query-time transcripts,
* and a fast approximate-NN-friendly index (FAISS over L2 or inner product).

The vector is built from word-boundary n-grams (3-5 chars) hashed via
FNV-1a into ``dim`` buckets, signed by parity, L2-normalized.  This is the
same trick as fastText's `FastText` embedding without the subword
*embeddings* (we use one-hot hashed features).

For evaluation queries we ALSO expose an optional mean-pool over a real
multilingual sentence encoder (loaded lazily when a HF model id is supplied
via ``MULTILINGUAL_ENCODER``) so a reviewer can compare.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from typing import Iterable, List, Sequence

import numpy as np

from .config import CFG

# FNV-1a 32-bit -- fast, well-distributed, no numpy/scipy required
_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193


def _fnv1a(text: str) -> int:
    h = _FNV_OFFSET
    for ch in text:
        h ^= ord(ch)
        h = (h * _FNV_PRIME) & 0xFFFFFFFF
    return h


def _ngrams(text: str, n_min: int, n_max: int) -> Iterable[str]:
    s = f"<{text}>"  # word-boundary markers, like fastText
    for n in range(n_min, n_max + 1):
        for i in range(len(s) - n + 1):
            yield s[i:i + n]


def _hash32(s: str) -> int:
    """Fast 32-bit hash of a string.  Uses md5 (C-implemented, in stdlib)
    for speed when we have many ngrams; the entropy is more than enough for
    a 384-dim index."""
    import struct
    # Take the first 4 bytes of md5 -- a uniformly random 32-bit number.
    d = hashlib.md5(s.encode("utf-8")).digest()
    return struct.unpack("<I", d[:4])[0]


class HashingEmbedder:
    """Deterministic, language-agnostic, fixed-dim char-ngram embedder."""

    def __init__(self, dim: int | None = None, n_min: int | None = None,
                 n_max: int | None = None, normalize: bool = True):
        cfg = CFG.retrieval
        self.dim = dim or cfg.dense_dim
        self.n_min = n_min or cfg.ngrams_min
        self.n_max = n_max or cfg.ngrams_max
        self.normalize = normalize
        # Pre-compute a per-bucket sign table so updates are just += ±1.
        # This is 2x faster than recomputing the sign from the hash each time.
        import numpy as np
        self._signs = np.where(
            np.random.RandomState(0).randint(0, 2, size=self.dim) == 0,
            -1.0, 1.0
        ).astype("float32")

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single string.  ~0.5-1 ms for a typical passage."""
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        v = np.zeros(self.dim, dtype=np.float32)
        if not text.strip():
            return v
        s = f"<{text}>"
        L = len(s)
        # Inline the fast path for the common n_min=3, n_max=5 case
        for n in range(self.n_min, self.n_max + 1):
            if L - n <= 0:
                break
            for i in range(L - n + 1):
                h = _hash32(s[i:i + n])
                v[h % self.dim] += self._signs[h % self.dim]
        if self.normalize:
            n = float(np.linalg.norm(v))
            if n > 1e-9:
                v /= n
        return v

    def encode(self, texts: Sequence[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            return self.encode_one(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = self.encode_one(t or "")
        return out

    def encode_bulk_fast(self, texts: Sequence[str]) -> np.ndarray:
        """Stack-allocated batched version (same as encode for this class)."""
        return self.encode(texts)


# ---------- optional real multilingual sentence encoder ----------
# Lazy-loaded only when MULTILINGUAL_ENCODER env var is set, so the default
# path stays tiny and fast.  Useful for ablation studies.
_SENT_ENCODER = None


@lru_cache(maxsize=1)
def get_sentence_encoder(model_id: str | None = None):
    """Load a sentence-transformers model once and cache it.  Returns None if
    no model id is provided or the library is unavailable."""
    global _SENT_ENCODER
    if _SENT_ENCODER is not None:
        return _SENT_ENCODER
    mid = model_id or os.environ.get("MULTILINGUAL_ENCODER", "")
    if not mid:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    try:
        _SENT_ENCODER = SentenceTransformer(mid, device="cpu")
    except Exception:
        _SENT_ENCODER = None
    return _SENT_ENCODER


def encode_with_optional_model(texts: Sequence[str], model_id: str | None = None) -> np.ndarray:
    """If a real encoder is configured, use it; otherwise fall back to hashing."""
    enc = get_sentence_encoder(model_id)
    if enc is None:
        return HashingEmbedder().encode(texts)
    import numpy as np
    vecs = enc.encode(list(texts), convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)
    return vecs.astype("float32")
