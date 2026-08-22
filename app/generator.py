"""Answer generation.

Two paths, always grounded in the retrieved passages:

1. *Extractive* (default, no network, ~1 ms): pick the retrieved sentence
   most aligned with the query, with a small surrounding context window.
2. *Generative* (when ``USE_LLM=1`` and ``SARVAM_API_KEY`` is set): call
   Sarvam ``sarvam-2`` with a strict grounding prompt.  A post-hoc
   hallucination check (token overlap with retrieved context) gates the
   response and falls back to the extractive answer if the generative
   answer is not adequately grounded.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Sequence

from .chunking import Chunk, split_sentences, tokenize
from .config import CFG
from .sarvam import llm_chat


@dataclass
class Answer:
    text: str
    mode: str            # "extractive" | "generative" | "refused" | "no_context"
    sources: list[dict]  # provenance for UI
    groundedness: float  # 0..1 lexical overlap with retrieved context
    latency_ms: float


# ---------------- extractive ----------------
# Stop words used to filter the "what is" / "how do" boilerplate so that
# the answer-vs-query overlap check focuses on the actual content.
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be",
    "to", "of", "in", "on", "at", "for", "by", "with",
    "and", "or", "but", "how", "what", "when", "where",
    "why", "who", "which", "do", "does", "did", "can",
    "could", "would", "should", "i", "you", "we", "they",
    "it", "this", "that", "these", "those", "about", "many",
    "much", "some", "any", "all", "no", "not", "than", "then",
}


def _extractive_answer(query: str, chunks: Sequence[Chunk], max_chars: int = 480) -> Answer:
    if not chunks:
        return Answer(text="", mode="no_context", sources=[], groundedness=0.0, latency_ms=0.0)
    q_tokens_all = set(tokenize(query))
    q_content = q_tokens_all - _STOP
    if not q_content:
        # Query is all stop words; fall back to a random sentence
        q_content = q_tokens_all

    # For each retrieved chunk, score every sentence by:
    #   1) content-word overlap with the query (the "is this on-topic" signal)
    #   2) length penalty (avoid 1-word and 100-word sentences)
    # Pick the sentence with the highest score, then return it with a
    # 1-sentence window around it for context.
    best: tuple[float, Chunk, int, str] | None = None
    for c in chunks:
        sents = split_sentences(c.text)
        for i, s in enumerate(sents):
            toks = set(tokenize(s))
            if not toks:
                continue
            s_content = toks & _STOP
            toks_content = toks - _STOP
            if not toks_content:
                continue
            overlap = len(toks_content & q_content) / len(q_content)
            # length sweet-spot: prefer 30-150 char sentences
            n_chars = len(s)
            len_pen = 1.0 if 30 <= n_chars <= 250 else 0.5
            score = overlap * len_pen
            if best is None or score > best[0]:
                best = (score, c, i, s)

    if not best:
        # No sentence had any content overlap -- fall back to the first
        # sentence of the top chunk.  The post-generation guardrail will
        # catch the lack of grounding.
        c = chunks[0]
        sents = split_sentences(c.text)
        s = sents[0] if sents else c.text[:max_chars]
        return Answer(text=s[:max_chars], mode="extractive",
                      sources=[_src(c)], groundedness=0.0, latency_ms=0.0)

    score, best_chunk, i, best_sent = best
    sents = split_sentences(best_chunk.text)
    window = sents[max(0, i - 1): i + 2]
    text = " ".join(window).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return Answer(
        text=text,
        mode="extractive",
        sources=[_src(best_chunk)],
        groundedness=_groundedness(text, [c.text for c in chunks[:3]]),
        latency_ms=0.0,
    )


# ---------------- generative (Sarvam LLM, opt-in) ----------------
GENERATION_SYSTEM_PROMPT = """You are a careful multilingual assistant.
You will be given a user question and a set of retrieved passages from a
knowledge base. Your job is to write a concise answer **in the same language
as the question**, using ONLY facts that appear in the passages. If the
passages do not contain the answer, reply exactly with: REFUSE.

Rules:
- 1-3 sentences, max 80 words.
- Do not invent entities, dates, numbers, or names.
- Do not introduce outside knowledge.
- If a fact is not in the passages, omit it.
- Always end with a brief "Source: <one-sentence fragment from a passage>" line."""


def _generative_answer(query: str, chunks: Sequence[Chunk], max_chars: int) -> Answer:
    t0 = time.time()
    if not CFG.generation.use_llm:
        return _extractive_answer(query, chunks, max_chars)
    context_blocks = []
    for i, c in enumerate(chunks[:5]):
        context_blocks.append(f"[{i+1}] {c.text}")
    user_msg = (f"Question: {query}\n\nPassages:\n" +
                "\n\n".join(context_blocks) +
                "\n\nAnswer concisely in the question's language.")
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    out = llm_chat(messages)
    elapsed = (time.time() - t0) * 1000
    text = (out.get("content") or "").strip()
    if not text or text.strip().upper().startswith("REFUSE"):
        # Model refused -- fall back to extractive so the user still gets something
        ans = _extractive_answer(query, chunks, max_chars)
        ans.latency_ms = elapsed
        return ans
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    g = _groundedness(text, [c.text for c in chunks[:3]])
    if g < CFG.generation.min_overlap:
        # Not grounded enough -- fall back to extractive
        ans = _extractive_answer(query, chunks, max_chars)
        ans.latency_ms = elapsed
        ans.mode = "extractive_fallback"
        return ans
    return Answer(
        text=text,
        mode="generative",
        sources=[_src(c) for c in chunks[:3]],
        groundedness=g,
        latency_ms=elapsed,
    )


# ---------------- helpers ----------------
def _src(c: Chunk) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "strategy": c.strategy,
        "snippet": c.short(200),
        "query_en": c.source_query_en,
        "target_lang": c.target_lang,
        "is_gold": c.is_gold,
    }


def _groundedness(answer: str, context_passages: Sequence[str]) -> float:
    """Quick lexical overlap: ratio of answer content tokens that appear in
    ANY retrieved passage.  Cheap, no model required, surprisingly
    effective at catching obvious hallucinations."""
    if not answer or not context_passages:
        return 0.0
    a = tokenize(answer)
    if not a:
        return 0.0
    ctx = set()
    for p in context_passages:
        ctx.update(tokenize(p))
    if not ctx:
        return 0.0
    a_set = set(a)
    return len(a_set & ctx) / len(a_set)


def generate(query: str, chunks: Sequence[Chunk]) -> Answer:
    """Public entry: choose generative if enabled, else extractive."""
    if CFG.generation.use_llm:
        return _generative_answer(query, chunks, CFG.generation.max_answer_chars)
    return _extractive_answer(query, chunks, CFG.generation.max_answer_chars)
