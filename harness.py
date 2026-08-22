"""The harness: structured orchestration around the RAG pipeline.

Responsibilities:

* Define a stable request/response schema (``QueryRequest`` /
  ``QueryResponse``) used by the HTTP server, the CLI, and tests.
* Run the pipeline in a fixed order with per-stage timing.
* Apply retries with exponential backoff for the network-bound stages
  (Sarvam STT/TTS/LLM).  Local stages are assumed transient-error-free.
* Surface guardrail refusals with structured reasons, never raise.
* Log every run to the latency telemetry ring buffer.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from . import guardrails, sarvam
from .chunking import Chunk
from .config import CFG
from .generator import Answer, generate
from .latency import LOG, StageTimer
from .retriever import HybridRetriever

log = logging.getLogger("harness")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


# ---------- schemas ----------
@dataclass
class QueryRequest:
    """All fields optional except `query`."""
    query: str | None = None          # if set, skip STT
    audio_b64: str | None = None      # if set, run STT first
    audio_content_type: str = "audio/wav"
    target_language: str | None = None   # BCP-47 hint; e.g. "hi-IN"
    top_k: int = 5
    want_tts: bool = False
    tts_speaker: str = "anushka"
    request_id: str = ""

    def validate(self) -> str | None:
        if not self.query and not self.audio_b64:
            return "either `query` or `audio_b64` must be provided"
        return None


@dataclass
class QueryResponse:
    ok: bool
    answer: str = ""
    answer_mode: str = ""        # extractive | generative | refused | no_context
    groundedness: float = 0.0
    transcript: str = ""
    detected_language: str = ""
    stt_confidence: float = 0.0
    sources: list[dict] = field(default_factory=list)
    refused: dict | None = None
    timings: dict = field(default_factory=dict)
    request_id: str = ""
    error: str = ""
    tts_audio_b64: str = ""      # base64-encoded WAV if TTS was requested

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- harness ----------
class Harness:
    """The single entry point every other component uses to answer a query."""

    def __init__(self, retriever: HybridRetriever | None = None):
        self.retriever = retriever

    def set_retriever(self, r: HybridRetriever):
        self.retriever = r

    # --- the main pipeline ---
    def run(self, req: QueryRequest) -> QueryResponse:
        t0 = time.perf_counter()
        timer = StageTimer(budget=CFG.latency.budget_total_ms)

        # 0) validate request
        v = req.validate()
        if v:
            return _err(req, f"validation_error: {v}", t0, timer)

        # 1) STT (or text)
        if req.query:
            transcript = req.query
            detected_lang = req.target_language or ""
            stt_conf = 1.0
            timer.add("stt_ms", 0.0)
        else:
            audio = _b64dec(req.audio_b64 or "")
            res = self._stt_with_retry(audio, req.audio_content_type, req.target_language)
            timer.add("stt_ms", _ms_since(t0, "stt"))
            if not res.text:
                return _err(req, "speech_to_text_failed: " + str(res.raw)[:200], t0, timer)
            transcript = res.text
            detected_lang = res.language or req.target_language or ""
            stt_conf = res.confidence

        # 2) greeting short-circuit
        greet = guardrails.greet_if_appropriate(transcript)
        if greet:
            timer.finish(t0)
            LOG.record(timer)
            return QueryResponse(
                ok=True, answer=greet, answer_mode="greeting",
                transcript=transcript, detected_language=detected_lang,
                stt_confidence=stt_conf, timings=timer.to_dict(),
                request_id=req.request_id,
            )

        # 3) text guardrails
        verdict = guardrails.guard_query(transcript)
        if not verdict.ok:
            timer.finish(t0)
            LOG.record(timer)
            return QueryResponse(
                ok=True, answer=verdict.reason, answer_mode="refused",
                refused={"code": verdict.code, "detail": verdict.detail},
                transcript=transcript, detected_language=detected_lang,
                stt_confidence=stt_conf, timings=timer.to_dict(),
                request_id=req.request_id,
            )

        # 4) retrieval
        if self.retriever is None:
            return _err(req, "no_index_loaded", t0, timer)
        t_ret = time.perf_counter()
        hits = self.retriever.retrieve(transcript, top_k=req.top_k)
        timer.add("retrieve_ms", (time.perf_counter() - t_ret) * 1000)

        # 5) retrieval guardrail
        v2 = guardrails.guard_retrieval(transcript, hits)
        if not v2.ok:
            timer.finish(t0)
            LOG.record(timer)
            return QueryResponse(
                ok=True, answer=v2.reason, answer_mode="refused",
                refused={"code": v2.code, "detail": v2.detail},
                transcript=transcript, detected_language=detected_lang,
                stt_confidence=stt_conf, timings=timer.to_dict(),
                request_id=req.request_id,
            )

        # 6) generate
        t_gen = time.perf_counter()
        chunks = [c for c, _s, _m in hits]
        ans: Answer = generate(transcript, chunks)
        timer.add("generate_ms", (time.perf_counter() - t_gen) * 1000)

        # 6b) post-generation groundedness check -- if the produced answer
        # is not grounded in any retrieved passage, refuse rather than ship
        # a hallucinated response.
        v3 = guardrails.guard_groundedness(ans.text, chunks,
                                           query=transcript,
                                           groundedness=ans.groundedness)
        if not v3.ok:
            timer.finish(t0)
            LOG.record(timer)
            return QueryResponse(
                ok=True, answer=v3.reason, answer_mode="refused",
                refused={"code": v3.code, "detail": v3.detail},
                transcript=transcript, detected_language=detected_lang,
                stt_confidence=stt_conf,
                timings=timer.to_dict(),
                request_id=req.request_id,
            )

        # 7) optional TTS
        tts_b64 = ""
        if req.want_tts and ans.text and CFG.sarvam.api_key:
            t_tts = time.perf_counter()
            tts_lang = _pick_tts_lang(detected_lang)
            tts = sarvam.tts_synthesize(ans.text, target_language=tts_lang,
                                        speaker=req.tts_speaker)
            timer.add("tts_ms", (time.perf_counter() - t_tts) * 1000)
            tts_b64 = tts.audio_b64

        # 8) finalize
        timer.finish(t0)
        LOG.record(timer)

        return QueryResponse(
            ok=True,
            answer=ans.text,
            answer_mode=ans.mode,
            groundedness=ans.groundedness,
            transcript=transcript,
            detected_language=detected_lang,
            stt_confidence=stt_conf,
            sources=ans.sources,
            timings=timer.to_dict(),
            request_id=req.request_id,
            error="",
            tts_audio_b64=tts_b64,
        )

    # --- internal: STT with retry ---
    def _stt_with_retry(self, audio: bytes, content_type: str, lang_hint: str | None,
                        max_attempts: int = 2) -> sarvam.STTResult:
        last = sarvam.STTResult(text="", language="", confidence=0.0, raw={})
        for attempt in range(max_attempts):
            last = sarvam.stt_transcribe(audio, content_type=content_type,
                                         language_hint=lang_hint)
            if last.text:
                return last
            time.sleep(0.1 * (2 ** attempt))
        return last


# ---------- helpers ----------
def _ms_since(t0: float, _tag: str) -> float:
    """Compat helper: milliseconds since perf_counter t0, no extra tag needed."""
    return (time.perf_counter() - t0) * 1000


def _pick_tts_lang(detected: str) -> str:
    """Map STT language code -> TTS language code.  Sarvam bulbul expects
    BCP-47 like 'hi-IN', 'en-IN', etc."""
    if not detected:
        return "hi-IN"
    d = detected.lower()
    if d.startswith("hi"): return "hi-IN"
    if d.startswith("en"): return "en-IN"
    if d.startswith("bn"): return "bn-IN"
    if d.startswith("mr"): return "mr-IN"
    if d.startswith("ta"): return "ta-IN"
    if d.startswith("te"): return "te-IN"
    if d.startswith("kn"): return "kn-IN"
    if d.startswith("ml"): return "ml-IN"
    if d.startswith("gu"): return "gu-IN"
    if d.startswith("pa"): return "pa-IN"
    if d.startswith("or"): return "od-IN"
    if d.startswith("as"): return "as-IN"
    return "hi-IN"


def _b64dec(s: str) -> bytes:
    import base64
    return base64.b64decode(s)


def _err(req: QueryRequest, msg: str, t0: float, timer: StageTimer) -> QueryResponse:
    timer.finish(t0)
    LOG.record(timer)
    return QueryResponse(ok=False, error=msg, request_id=req.request_id,
                         timings=timer.to_dict())
