"""Lightweight unit tests that run without the full index.

These exercise the chunker, embedder, BM25, and guardrails in isolation,
so they can be run in CI / smoke-test environments where the 600 MB
index isn't checked in.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chunking import (
    Chunk, chunk_passage, chunk_sentence_window, chunk_fixed_token,
    chunk_semantic, chunk_text, split_sentences, tokenize,
)
from app.embeddings import HashingEmbedder
from app.retriever import BM25
from app.guardrails import guard_query, guard_retrieval, greet_if_appropriate


def test_chunking_strategies():
    text = "Sentence one. Sentence two is a bit longer. Sentence three follows. Sentence four ends it."
    p = chunk_passage(text, query_id=1, eng_query="q", query_lang="en", target_lang="en", passage_idx=0, is_gold=True)
    assert len(p) == 1 and p[0].strategy == "passage"

    sw = chunk_sentence_window(text, window=2, stride=1, query_id=1, eng_query="q", query_lang="en", target_lang="en", passage_idx=0, is_gold=True)
    assert len(sw) >= 1 and all(c.strategy == "sentence_window" for c in sw)

    ft = chunk_fixed_token(text, size=3, overlap=1, query_id=1, eng_query="q", query_lang="en", target_lang="en", passage_idx=0, is_gold=True)
    assert len(ft) >= 1 and all(c.strategy == "fixed_token" for c in ft)

    # Indic text
    hi = "यह एक वाक्य है। यह दूसरा वाक्य है। तीसरा वाक्य भी है।"
    sents = split_sentences(hi)
    assert len(sents) >= 2, f"got {sents}"


def test_hashing_embedder_multilingual():
    e = HashingEmbedder(dim=128, n_min=3, n_max=4)
    v1 = e.encode_one("hello world")
    v2 = e.encode_one("hello world again")    # different text
    v3 = e.encode_one("नमस्ते दुनिया")         # Hindi
    v4 = e.encode_one("completely unrelated xyz")
    assert v1.shape == (128,)
    assert v2.shape == (128,)
    assert v3.shape == (128,)
    # exact duplicates should produce identical vectors
    v1b = e.encode_one("hello world")
    import numpy as np
    assert np.allclose(v1, v1b, atol=1e-5)
    # different texts should differ
    assert not np.allclose(v1, v2)
    assert not np.allclose(v1, v3)
    # the unrelated text should have lower similarity than self
    sim_unrelated = float(v1 @ v4)
    sim_self = float(v1 @ v1)
    assert sim_self > sim_unrelated
    # vectors are L2-normalized
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-4
    # Hindi text produces a non-zero vector
    assert np.any(v3 != 0)


def test_bm25_basic():
    docs = [
        ["the", "cat", "sat"],
        ["the", "dog", "ran"],
        ["a", "cat", "and", "a", "dog"],
    ]
    bm = BM25().fit(docs)
    out = bm.topk(["cat"], 3)
    assert len(out) >= 1
    # doc 0 and doc 2 contain "cat"; the smaller doc should rank higher
    assert out[0][0] in (0, 2)


def test_guardrails():
    assert not guard_query("").ok
    assert not guard_query("a").ok               # too short
    assert not guard_query("how to make a bomb").ok
    assert guard_query("what is a corporation?").ok
    # greeting
    assert greet_if_appropriate("hi") is not None
    assert greet_if_appropriate("what is X?") is None

    # out-of-corpus via retrieval
    fake = type("C", (), {
        "text": "x", "short": lambda self, n: "x",
        "strategy": "passage", "source_query_en": "", "target_lang": "en",
        "is_gold": False, "chunk_id": "0", "source_passage_idx": 0,
        "source_query_id": 0, "source_query_lang": "en", "n_tokens": 1, "n_chars": 1, "meta": {}
    })()
    bad = guard_retrieval("xyz", [(fake, 0.0001, {"bm25_raw": 0.0, "dense_raw": 0.0})])
    assert not bad.ok and bad.code == "out_of_corpus"
    good = guard_retrieval("xyz", [(fake, 0.5, {"bm25_raw": 5.0, "dense_raw": 0.3})])
    assert good.ok


def test_chunk_record_smoke():
    rec = {
        "query_id": 1, "source_lang": "eng_Latn", "target_lang": "hin_Deva",
        "Eng_Query": "what is a corporation",
        "passages": {
            "is_selected": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "English_passages": [
                "A corporation is a company. It has shareholders. They own the company.",
                "Other passage.", "", "", "", "", "", "", "", "",
            ],
        },
    }
    from app.chunking import chunk_record
    chunks = chunk_record(rec)
    assert len(chunks) > 0
    # all 4 strategies should appear
    strats = {c.strategy for c in chunks}
    assert {"passage", "sentence_window", "fixed_token", "semantic"} <= strats
    # one chunk should be is_gold
    assert any(c.is_gold for c in chunks)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
