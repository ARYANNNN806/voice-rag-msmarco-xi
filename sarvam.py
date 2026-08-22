"""Sarvam AI client wrappers.

* Speech-to-text (saarika:v2.5) -- POST audio file, get transcript + lang
* Text-to-speech (bulbul:v2)    -- POST text, get WAV bytes
* Chat completions (sarvam-2)    -- optional, for generative answers

All three are best-effort and time-bounded; failures are non-fatal and the
pipeline degrades gracefully (e.g. falls back to a known-good transcript
or to an extractive answer).
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import CFG, SarvamConfig


# language detection hints -- Sarvam STT supports these BCP-47 codes
SARVAM_SUPPORTED = {
    "hi-IN", "en-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN",
    "pa-IN", "ta-IN", "te-IN", "gu-IN", "as-IN",
}


@dataclass
class STTResult:
    text: str
    language: str            # BCP-47
    confidence: float        # 0..1
    raw: dict


@dataclass
class TTSResult:
    audio_b64: str
    format: str              # "wav"
    raw: dict


# --------------------- STT ---------------------
def stt_transcribe(audio_bytes: bytes, *, content_type: str = "audio/wav",
                   cfg: SarvamConfig | None = None,
                   language_hint: str | None = None,
                   timeout_s: float | None = None) -> STTResult:
    """Transcribe an audio blob via Sarvam saarika:v2.5.

    ``content_type`` should be one of the formats Sarvam accepts: wav, mp3,
    webm, ogg, m4a.  ``language_hint`` (BCP-47) is passed when the caller
    already knows the language; otherwise Sarvam auto-detects.
    """
    cfg = cfg or CFG.sarvam
    if not cfg.api_key:
        return STTResult(text="", language="", confidence=0.0, raw={"error": "no_api_key"})
    headers = {"api-subscription-key": cfg.api_key}
    files = {"file": ("audio." + content_type.split("/")[-1], audio_bytes, content_type)}
    data = {"model": cfg.stt_model, "mode": "transcribe", "with_diarization": "false"}
    if language_hint and language_hint in SARVAM_SUPPORTED:
        data["language_code"] = language_hint

    t0 = time.time()
    try:
        with httpx.Client(timeout=timeout_s or cfg.timeout_s) as client:
            r = client.post(cfg.stt_endpoint, headers=headers, files=files, data=data)
        elapsed = (time.time() - t0) * 1000
        if r.status_code != 200:
            return STTResult(text="", language="", confidence=0.0,
                             raw={"status": r.status_code, "body": r.text, "elapsed_ms": elapsed})
        body = r.json()
    except Exception as e:
        return STTResult(text="", language="", confidence=0.0,
                         raw={"error": str(e)})

    text = (body.get("transcript") or "").strip()
    lang = body.get("language_code") or body.get("language") or (language_hint or "")
    conf = float(body.get("confidence", 0.0))
    return STTResult(text=text, language=lang, confidence=conf, raw=body)


# --------------------- TTS ---------------------
def tts_synthesize(text: str, *, target_language: str = "hi-IN",
                   speaker: str = "anushka", cfg: SarvamConfig | None = None,
                   timeout_s: float | None = None) -> TTSResult:
    """Synthesize speech with Sarvam bulbul:v2.  Returns base64-encoded WAV."""
    cfg = cfg or CFG.sarvam
    if not cfg.api_key:
        return TTSResult(audio_b64="", format="wav", raw={"error": "no_api_key"})
    headers = {"api-subscription-key": cfg.api_key, "Content-Type": "application/json"}
    payload = {
        "inputs": [text[:1500]],
        "target_language_code": target_language,
        "speaker": speaker,
        "model": cfg.tts_model,
        "pace": 1.0,
        "speech_sample_rate": 22050,
        "output_audio_codec": "wav",
    }
    try:
        with httpx.Client(timeout=timeout_s or cfg.timeout_s) as client:
            r = client.post(cfg.tts_endpoint, headers=headers, json=payload)
        if r.status_code != 200:
            return TTSResult(audio_b64="", format="wav",
                             raw={"status": r.status_code, "body": r.text})
        body = r.json()
    except Exception as e:
        return TTSResult(audio_b64="", format="wav", raw={"error": str(e)})

    audios = body.get("audios") or []
    if not audios:
        return TTSResult(audio_b64="", format="wav", raw=body)
    return TTSResult(audio_b64=audios[0], format="wav", raw=body)


# --------------------- LLM (optional) ---------------------
def llm_chat(messages: list[dict], *, cfg: SarvamConfig | None = None,
             timeout_s: float | None = None) -> dict[str, Any]:
    """One-shot chat with sarvam-2.  Used for generative answers.

    Returns a dict with ``content`` (str) and ``raw``.  Errors are returned
    in ``raw`` rather than raised -- the harness handles them.
    """
    cfg = cfg or CFG.sarvam
    if not cfg.api_key:
        return {"content": "", "raw": {"error": "no_api_key"}}
    headers = {"api-subscription-key": cfg.api_key, "Content-Type": "application/json"}
    payload = {"model": cfg.llm_model, "messages": messages, "temperature": 0.2,
               "max_tokens": 350}
    try:
        with httpx.Client(timeout=timeout_s or cfg.timeout_s) as client:
            r = client.post(cfg.llm_endpoint, headers=headers, json=payload)
        if r.status_code != 200:
            return {"content": "", "raw": {"status": r.status_code, "body": r.text}}
        body = r.json()
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"content": content, "raw": body}
    except Exception as e:
        return {"content": "", "raw": {"error": str(e)}}


# --------------------- helpers ---------------------
def audio_b64_to_bytes(b: str) -> bytes:
    """Decode a Sarvam TTS base64 string."""
    return base64.b64decode(b)
