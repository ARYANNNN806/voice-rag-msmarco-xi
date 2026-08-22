"""Interactive voice/text demo.

Usage::

    python -m scripts.cli            # text-only REPL (no Sarvam needed)
    python -m scripts.cli --voice    # record voice via mic, transcribe, answer, TTS

If ``SARVAM_API_KEY`` is set, ``--voice`` records from the default mic,
sends the audio to Sarvam saarika:v2.5 for STT, runs the RAG pipeline,
and reads the answer back with Sarvam bulbul:v2 TTS.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.harness import Harness, QueryRequest
from app.retriever import HybridRetriever


def text_loop(h: Harness):
    print("text mode.  type a question and hit enter (empty to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        out = h.run(QueryRequest(query=q, want_tts=False))
        _print_out(out)


def _print_out(out):
    print()
    print(f"  answer_mode: {out.answer_mode}")
    print(f"  transcript : {out.transcript!r}")
    print(f"  lang       : {out.detected_language}")
    print(f"  grounded   : {out.groundedness:.3f}")
    print(f"  timings    : total={out.timings.get('total_ms', 0):.1f}ms  "
          + " ".join(f"{k}={v:.1f}ms" for k, v in out.timings.get("stages", {}).items()))
    print()
    print("  " + (out.answer or "").replace("\n", "\n  "))
    if out.refused:
        print(f"  [refused: {out.refused.get('code')}]")


def voice_loop(h: Harness):
    from app.sarvam import stt_transcribe, tts_synthesize, audio_b64_to_bytes
    import sounddevice as sd
    import soundfile as sf
    import numpy as np

    sr = 16000
    seconds = 5
    print("voice mode: press ENTER to record, ENTER again to stop, 'q' to quit.")
    while True:
        cmd = input("\n[rec?] ").strip().lower()
        if cmd in ("q", "quit", "exit"):
            return
        print(f"  recording {seconds}s @ {sr} Hz -- speak now…")
        audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
        sd.wait()
        # Save to WAV
        tmp = "/tmp/voice_rag_query.wav"
        sf.write(tmp, audio, sr)
        with open(tmp, "rb") as f:
            blob = f.read()

        t = time.time()
        stt = stt_transcribe(blob, content_type="audio/wav")
        print(f"  STT: {stt.text!r}  (lang={stt.language} conf={stt.confidence:.2f}, {time.time()-t:.1f}s)")
        if not stt.text.strip():
            print("  (empty transcript, try again)")
            continue

        out = h.run(QueryRequest(query=stt.text, target_language=stt.language))
        _print_out(out)

        if out.answer:
            tts_lang = (stt.language or "hi-IN").split("-")[0] + "-IN"
            tts = tts_synthesize(out.answer, target_language=tts_lang)
            if tts.audio_b64:
                wav = audio_b64_to_bytes(tts.audio_b64)
                play_path = "/tmp/voice_rag_answer.wav"
                with open(play_path, "wb") as f:
                    f.write(wav)
                data, _ = sf.read(play_path)
                sd.play(data, 22050)
                sd.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", action="store_true")
    args = ap.parse_args()

    if not (ROOT / "index" / "chunks.jsonl").exists():
        print("Index not found. Run: python -m scripts.build_index", file=sys.stderr)
        sys.exit(1)
    r = HybridRetriever.load()
    h = Harness(retriever=r)
    print(f"index: {len(r.chunks)} chunks across "
          f"{len({c.target_lang for c in r.chunks if c.target_lang})} languages")
    if args.voice:
        if not os.environ.get("SARVAM_API_KEY"):
            print("SARVAM_API_KEY not set; voice mode requires Sarvam.", file=sys.stderr)
            sys.exit(1)
        voice_loop(h)
    else:
        text_loop(h)


if __name__ == "__main__":
    main()
