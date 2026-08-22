"""Configuration for the voice RAG pipeline.

All runtime parameters are env-overridable so the same image can be tuned
without rebuilding.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------- paths ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
WEB_DIR = ROOT / "web"
INDEX_DIR.mkdir(exist_ok=True)


# ---------- sarvam ----------
@dataclass
class SarvamConfig:
    """Settings for the Sarvam AI APIs (STT / TTS / optional LLM)."""
    api_key: str = os.environ.get("SARVAM_API_KEY", "")
    # saarika:v2 supports 11 Indic languages + English; v2.5 adds more.
    stt_model: str = os.environ.get("SARVAM_STT_MODEL", "saarika:v2.5")
    stt_endpoint: str = "https://api.sarvam.ai/speech-to-text"
    # bulbul:v2 for natural-sounding Indic TTS
    tts_model: str = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v2")
    tts_endpoint: str = "https://api.sarvam.ai/text-to-speech"
    # sarvam-2 is the LLM (used for generative answers when enabled)
    llm_model: str = os.environ.get("SARVAM_LLM_MODEL", "sarvam-2")
    llm_endpoint: str = "https://api.sarvam.ai/v1/chat/completions"
    timeout_s: float = float(os.environ.get("SARVAM_TIMEOUT_S", "20"))


# ---------- retrieval ----------
@dataclass
class RetrievalConfig:
    # Hybrid: combine BM25 (lexical) with a multilingual char-ngram dense embedding.
    # This gives broad recall across all 14 Indic languages without a transformer.
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    dense_dim: int = 384            # hashing trick projection
    ngrams_min: int = 3
    ngrams_max: int = 5
    top_k_dense: int = 50
    top_k_bm25: int = 50
    top_k_final: int = 5
    rrf_k: int = 60                # reciprocal rank fusion constant
    dense_weight: float = 0.55
    bm25_weight: float = 0.45


# ---------- generation ----------
@dataclass
class GenerationConfig:
    # Two paths: extractive (always available, ~5ms) and generative (Sarvam LLM).
    use_llm: bool = os.environ.get("USE_LLM", "0") == "1"
    max_answer_chars: int = 480
    # Refusal heuristic: refuse if the *RRF* top-1 score is below this.
    # RRF scores are small (1/(k+rank) ~ 0.016 for top-1) so the threshold
    # is tiny.  Combined with a hard rule "refuse if both raw channels are
    # exactly zero" (handled in guardrails.guard_retrieval), this catches
    # out-of-corpus queries without false positives.
    refusal_score: float = float(os.environ.get("REFUSAL_SCORE", "0.005"))
    # Hallucination guard: the generated answer must overlap lexically with
    # at least one retrieved passage.
    min_overlap: float = float(os.environ.get("MIN_OVERLAP", "0.15"))


# ---------- latency budget ----------
@dataclass
class LatencyConfig:
    # All in milliseconds; the full pipeline (STT excluded -- it goes over the wire
    # and is reported separately) must fit in this budget.
    budget_total_ms: float = 200.0
    # Per-stage soft budgets used for telemetry
    budget_chunk_ms: float = 30.0
    budget_embed_ms: float = 40.0
    budget_retrieve_ms: float = 80.0
    budget_generate_ms: float = 40.0
    budget_guard_ms: float = 10.0


@dataclass
class AppConfig:
    sarvam: SarvamConfig = field(default_factory=SarvamConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)


CFG = AppConfig()
