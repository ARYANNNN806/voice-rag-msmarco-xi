"""Guardrails: detect and refuse off-topic / unsafe / unanswerable queries.

Four checks, in order:

1. **Length / empty** -- must be at least 2 tokens of substance.
2. **Language sanity** -- must contain word characters, not pure noise.
3. **Unsafe content** -- small blocklist of obviously disallowed topics
   (PII probes, instructions for wrongdoing).  This is intentionally
   conservative; it errs on the side of refusing and logs the trigger.
4. **Out-of-corpus** -- if the top retrieval score is below a threshold
   AND no retrieved chunk passes a "relevant" heuristic, refuse.

A refusal returns a structured ``Refusal`` object so the harness can show
the user a clear reason rather than guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .chunking import Chunk, tokenize
from .config import CFG

# Patterns we will refuse outright.  These are deliberately conservative
# and the list is intentionally short to avoid over-blocking.
_UNSAFE_PATTERNS = [
    r"\b(how\s+to\s+(make|build|synthesi[sz]e)\s+(a\s+)?(bomb|weapon|poison|drug))\b",
    r"\b(credit\s*card\s*(number|cvv))\b",
    r"\b(password|otp)\s+(of|for)\s+\w+",
    r"\bhow\s+to\s+(hack|exploit|ddos)\b",
    r"\b(child|minor)\s+(porn|abuse|exploit)\b",
]
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)

# Greetings / pleasantries that are valid user inputs but not retrievable
# questions.  We turn these into friendly small-talk responses.
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|namaste|namaskar|hola|salam|kem\s+cho|vanakkam|"
    r"adaab|sat\s+sri\s+akal|sup|yo)\b",
    re.IGNORECASE,
)


@dataclass
class GuardrailVerdict:
    ok: bool
    reason: str = ""              # human-readable
    code: str = "ok"              # machine-readable
    detail: dict | None = None


# ---------------- individual checks ----------------
def _check_language(text: str) -> GuardrailVerdict:
    if not text or not text.strip():
        return GuardrailVerdict(False, "I didn't catch any words. Please try again.",
                                "empty_input")
    # at least 2 word tokens (any script)
    toks = tokenize(text)
    if len(toks) < 2:
        return GuardrailVerdict(False,
                                "Your question was too short. Could you say a little more?",
                                "too_short")
    return GuardrailVerdict(True)


def _check_unsafe(text: str) -> GuardrailVerdict:
    m = _UNSAFE_RE.search(text)
    if m:
        return GuardrailVerdict(
            False,
            "I can't help with that. Please ask a question about the indexed knowledge base.",
            "unsafe_content", {"matched": m.group(0)})
    return GuardrailVerdict(True)


def _is_greeting(text: str) -> bool:
    return bool(_GREETING_RE.match(text.strip()))


# ---------------- public API ----------------
def guard_query(text: str) -> GuardrailVerdict:
    """Pure-text guardrails (do not depend on retrieval)."""
    v = _check_language(text)
    if not v.ok:
        return v
    return _check_unsafe(text)


def guard_retrieval(text: str, retrieved: list[tuple[Chunk, float, dict]]) -> GuardrailVerdict:
    """After retrieval: if scores are too low, refuse to answer.

    Two ways to fail:
      * no results at all (no_index)
      * top-1 RRF below the configured threshold AND both raw channels
        have negligible score -- this catches both empty corpora and queries
        with no lexical or semantic overlap.
    """
    if not retrieved:
        return GuardrailVerdict(False,
            "I don't have any indexed content to search. Try again after the index loads.",
            "no_index")
    top = retrieved[0]
    rrf_score = float(top[1])
    bm_raw = float(top[2].get("bm25_raw", 0.0))
    de_raw = float(top[2].get("dense_raw", 0.0))
    if rrf_score < CFG.generation.refusal_score and bm_raw < 0.5 and de_raw < 0.2:
        return GuardrailVerdict(
            False,
            "I couldn't find anything in the knowledge base that answers this. "
            "Try rephrasing or ask a more specific question.",
            "out_of_corpus",
            {"rrf_score": rrf_score, "bm25_top": bm_raw,
             "dense_top": de_raw, "threshold": CFG.generation.refusal_score})
    return GuardrailVerdict(True)


def guard_groundedness(answer: str, retrieved: Sequence[Chunk],
                       query: str = "", groundedness: float = 0.0) -> GuardrailVerdict:
    """After generation: if the answer is not grounded in either the
    retrieved context or the query, refuse.

    Two complementary signals:

    * **Answer-vs-query** token overlap (Jaccard on content words): a
      good answer should share at least *one* content word with the
      question.  We use a tiny threshold (0.0 = no overlap required)
      because the extractive answer may legitimately rephrase the
      question.  We only refuse on the much stronger "answer shares
      literally zero content words with the query" signal.

    * **Answer-vs-context** token overlap: the answer's words should
      come from the retrieved passages, not from outside.  This catches
      LLM hallucinations when the generative path is on.

    The original generator already does its own self-check on
    `groundedness` (answer ∩ query); this guardrail adds the
    answer-∉-context check and a much-stricter "answer is completely
    unrelated to the query" check.
    """
    if not answer or not retrieved:
        return GuardrailVerdict(True)

    a = set(tokenize(answer))
    if not a:
        return GuardrailVerdict(True)

    # 1) Context check: are the answer's words in the retrieved passages?
    ctx: set[str] = set()
    for c in retrieved:
        ctx.update(tokenize(c.text))
    if not ctx:
        return GuardrailVerdict(True)
    ctx_overlap = len(a & ctx) / len(a)
    if ctx_overlap < 0.5:
        return GuardrailVerdict(
            False,
            "I can't find a confident answer in the indexed content. "
            "Could you rephrase or ask a more specific question?",
            "low_context_overlap",
            {"ctx_overlap": round(ctx_overlap, 3), "min_required": 0.5})

    # 2) Query check: the answer's content words should overlap with the
    # question's content words.  We use a soft threshold: refuse only if
    # the Jaccard similarity is essentially 0 (i.e. zero shared content
    # words at all), which is the signature of "the model picked a
    # random sentence from a random passage."
    if query:
        q = set(tokenize(query))
        if q:
            stop = {"a", "an", "the", "is", "are", "was", "were", "be",
                    "to", "of", "in", "on", "at", "for", "by", "with",
                    "and", "or", "but", "how", "what", "when", "where",
                    "why", "who", "which", "do", "does", "did", "can",
                    "could", "would", "should", "i", "you", "we", "they",
                    "it", "this", "that", "these", "those", "about"}
            q_content = q - stop
            a_content = a - stop
            if q_content and a_content:
                # Jaccard
                jacc = len(a_content & q_content) / len(a_content | q_content)
                if jacc == 0.0:
                    return GuardrailVerdict(
                        False,
                        "I can't find a confident answer in the indexed content. "
                        "Could you rephrase or ask a more specific question?",
                        "low_query_overlap",
                        {"jaccard": 0.0, "q_content": sorted(q_content)[:10]})

    return GuardrailVerdict(True)


def greet_if_appropriate(text: str) -> str | None:
    """Return a canned greeting if the input looks like a greeting."""
    if _is_greeting(text):
        return ("Hello! I'm a voice-enabled RAG assistant over MSMARCO-XI. "
                "Ask me a question in any of 14 Indic languages or in English "
                "and I'll answer from the indexed passages.")
    return None
