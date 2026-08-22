"""FastAPI server exposing the RAG pipeline.

Endpoints:

* ``GET  /``              -- web UI
* ``GET  /healthz``       -- liveness + index info
* ``GET  /stats``         -- live latency P50/P70/P100 telemetry
* ``POST /query``         -- structured RAG query (text in, text out)
* ``POST /voice``         -- voice query (audio in, text + optional audio out)
* ``WS   /ws/voice``      -- streaming voice (browser mic -> STT -> answer -> TTS)

All endpoints return JSON with the same ``QueryResponse`` shape as the
harness, so the UI can render results consistently.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import sarvam
from .config import CFG, WEB_DIR
from .harness import Harness, QueryRequest, QueryResponse
from .latency import LOG
from .retriever import HybridRetriever

log = logging.getLogger("server")


# ---- app state ----
class State:
    harness: Harness | None = None
    retriever: HybridRetriever | None = None
    index_info: dict = {}


STATE = State()

app = FastAPI(title="Voice-Enabled RAG over MSMARCO-XI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup():
    """Load the index if one exists; otherwise build it on demand."""
    from .config import INDEX_DIR
    if (INDEX_DIR / "chunks.jsonl").exists():
        log.info("loading existing index from %s", INDEX_DIR)
        try:
            r = HybridRetriever.load(index_dir=INDEX_DIR)
            STATE.retriever = r
            STATE.harness = Harness(retriever=r)
            STATE.index_info = {
                "loaded": True,
                "n_chunks": len(r.chunks),
                "strategies": sorted({c.strategy for c in r.chunks}),
                "langs": sorted({c.target_lang for c in r.chunks if c.target_lang})[:20],
            }
        except Exception as e:
            log.exception("index load failed: %s", e)
            STATE.harness = Harness()
    else:
        log.warning("no index found at %s; /query will return 503 until built", INDEX_DIR)
        STATE.harness = Harness()


# ---- static UI ----
@app.get("/")
def root():
    p = WEB_DIR / "index.html"
    if not p.exists():
        return JSONResponse({"error": "UI not built"}, status_code=500)
    return FileResponse(p)


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ---- health & stats ----
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "index": STATE.index_info,
        "sarvam_configured": bool(CFG.sarvam.api_key),
        "use_llm": CFG.generation.use_llm,
        "budget_ms": CFG.latency.budget_total_ms,
    }


@app.get("/stats")
def stats():
    s = LOG.stats()
    s["budget_ms"] = CFG.latency.budget_total_ms
    s["budget_breach_pct"] = round(100.0 * s.get("budget_breaches", 0) / max(1, s.get("n", 1)), 2)
    return s


# ---- text query ----
class QueryIn(BaseModel):
    query: str
    target_language: str | None = None
    top_k: int = 5
    want_tts: bool = False
    tts_speaker: str = "anushka"
    request_id: str = ""


@app.post("/query")
def post_query(in_: QueryIn):
    if not STATE.harness or not STATE.retriever:
        raise HTTPException(503, "Index not loaded. POST /rebuild first or run scripts/build_index.py.")
    req = QueryRequest(**in_.model_dump())
    out: QueryResponse = STATE.harness.run(req)
    payload = out.to_dict()
    return JSONResponse(payload)


# ---- voice query (single shot) ----
@app.post("/voice")
async def post_voice(
    audio: UploadFile = File(...),
    target_language: str | None = Form(None),
    want_tts: bool = Form(False),
    tts_speaker: str = Form("anushka"),
    top_k: int = Form(5),
    request_id: str = Form(""),
):
    if not STATE.harness:
        raise HTTPException(503, "harness not ready")
    blob = await audio.read()
    audio_b64 = base64.b64encode(blob).decode("ascii")
    content_type = audio.content_type or "audio/wav"
    req = QueryRequest(
        audio_b64=audio_b64,
        audio_content_type=content_type,
        target_language=target_language,
        top_k=top_k,
        want_tts=want_tts,
        tts_speaker=tts_speaker,
        request_id=request_id,
    )
    out = STATE.harness.run(req)
    return JSONResponse(out.to_dict())


# ---- websocket voice (browser <-> server) ----
@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    await ws.accept()
    if not STATE.harness:
        await ws.send_json({"type": "error", "error": "harness_not_ready"})
        await ws.close()
        return
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"] is not None:
                # raw audio frame
                blob = msg["bytes"]
                audio_b64 = base64.b64encode(blob).decode("ascii")
                req = QueryRequest(
                    audio_b64=audio_b64,
                    audio_content_type="audio/wav",
                    want_tts=True,
                )
                out = STATE.harness.run(req)
                # send the response back (text + TTS)
                await ws.send_json({
                    "type": "response",
                    "ok": out.ok,
                    "answer": out.answer,
                    "answer_mode": out.answer_mode,
                    "transcript": out.transcript,
                    "detected_language": out.detected_language,
                    "stt_confidence": out.stt_confidence,
                    "groundedness": out.groundedness,
                    "sources": out.sources,
                    "refused": out.refused,
                    "timings": out.timings,
                    "error": out.error,
                })
            elif "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    data = {}
                if data.get("type") == "query":
                    req = QueryRequest(**{k: v for k, v in data.items() if k != "type"})
                    out = STATE.harness.run(req)
                    await ws.send_json({"type": "response", **out.to_dict()})
                elif data.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass


# ---- admin: rebuild index ----
@app.post("/rebuild")
def rebuild():
    """Trigger an asynchronous rebuild of the index from data/*.jsonl."""
    import subprocess, sys
    cmd = [sys.executable, "-m", "scripts.build_index"]
    proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent.parent))
    return {"started": True, "pid": proc.pid}
