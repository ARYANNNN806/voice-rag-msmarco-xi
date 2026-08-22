"""Live voice demo: record from mic, transcribe, answer, play back.

This is the script that produces the end-to-end "voice in, voice out"
experience.  Requires:

* A working microphone
* ``SARVAM_API_KEY`` for STT and TTS
* The FastAPI server running on http://localhost:8000

Run with::

    export SARVAM_API_KEY=...
    python -m scripts.record_demo
"""
from __future__ import annotations

import base64
import os
import sys
import time
import wave
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

API = os.environ.get("VOICE_RAG_API", "http://localhost:8000")


def record(seconds: float = 5.0, sr: int = 16000, path: str = "/tmp/voice_rag_q.wav") -> bytes:
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print("install sounddevice and soundfile for the voice demo", file=sys.stderr)
        sys.exit(1)
    print(f"  recording {seconds}s @ {sr} Hz…")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    sf.write(path, audio, sr)
    with open(path, "rb") as f:
        return f.read()


def transcribe(blob: bytes) -> dict:
    files = {"audio": ("q.wav", blob, "audio/wav")}
    r = requests.post(f"{API}/voice", files=files,
                      data={"want_tts": "true"}, timeout=60)
    r.raise_for_status()
    return r.json()


def play_wav_bytes(wav: bytes):
    try:
        import sounddevice as sd
        import soundfile as sf
        import io
    except ImportError:
        return
    data, sr = sf.read(io.BytesIO(wav))
    sd.play(data, sr)
    sd.wait()


def main():
    if not os.environ.get("SARVAM_API_KEY"):
        print("SARVAM_API_KEY is not set; the demo needs it for STT/TTS.",
              file=sys.stderr)
        sys.exit(1)

    print("Voice RAG demo.  Press Enter to record, 'q' to quit.\n")
    while True:
        cmd = input("[rec?] ").strip().lower()
        if cmd in ("q", "quit", "exit"):
            return
        t = time.time()
        blob = record()
        print(f"  recorded {len(blob)}B in {time.time()-t:.1f}s")
        t = time.time()
        try:
            out = transcribe(blob)
        except Exception as e:
            print(f"  request failed: {e}")
            continue
        print(f"  /voice responded in {time.time()-t:.1f}s")
        print(f"  transcript : {out.get('transcript')!r}")
        print(f"  lang       : {out.get('detected_language')}")
        print(f"  mode       : {out.get('answer_mode')}")
        print(f"  timings    : {out.get('timings', {}).get('total_ms', 0):.1f} ms")
        print(f"  answer     : {(out.get('answer') or '')[:200]!r}")
        if out.get("refused"):
            print(f"  refused    : {out['refused']}")
        # If Sarvam TTS was returned (currently we don't include it in the response;
        # this is where it would play), play it.
        # The current /voice endpoint does not return TTS audio by default; the
        # websocket endpoint does.  For now, this is a no-op if no audio.
        audio = out.get("tts_audio_b64")
        if audio:
            play_wav_bytes(base64.b64decode(audio))
        print()


if __name__ == "__main__":
    main()
