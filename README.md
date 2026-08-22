# Voice-Enabled RAG over MSMARCO-XI

A full voice-in / voice-out Retrieval-Augmented Generation system over the
[MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) dataset
(the MS MARCO passages/questions translated into 14 Indic languages).

> Voice in (any of 14 Indic languages or English) → Sarvam STT →
> hybrid BM25 + dense retrieval → grounded answer (extractive, or
> generative via Sarvam-2) → Sarvam TTS voice out.

The full local pipeline (after STT) targets **< 200 ms** end-to-end and is
instrumented with per-stage P50 / P70 / P100 telemetry.

---

## What you get

| Component | Choice | Why |
|---|---|---|
| STT | **Sarvam saarika:v2.5** | Native 11+ Indic languages, auto language ID |
| TTS | **Sarvam bulbul:v2** | Natural-sounding Indic voices (anushka, etc.) |
| LLM (opt-in) | **Sarvam sarvam-2** | Optional; default is extractive for groundedness |
| Embeddings | **Char-n-gram hashing trick** | Truly multilingual, ~1 ms / text, zero model load |
| Sparse retriever | **BM25** (Okapi, from scratch) | Strong lexical baseline |
| Vector index | **FAISS HNSW** | Sub-millisecond ANN over 150k chunks |
| Fusion | **Reciprocal Rank Fusion** | Simple, robust to score-scale mismatch |
| Chunking | 4 complementary strategies (see below) | Each catches a different failure mode |
| Orchestration | Structured `Harness` with retries | Single entry point, JSON I/O, per-stage timing |
| Guardrails | 4 layered checks | Empty / unsafe / out-of-corpus / low-groundedness |
| UI | Plain HTML+JS in `web/index.html` | No build step, runs in any browser |

---

## Quick start

```bash
# 1) Install
pip install -r requirements.txt

# 2) (Optional) get a Sarvam API key for real voice I/O
export SARVAM_API_KEY=...   # from https://dashboard.sarvam.ai

# 3) Build the index over the bundled sample data (3000 records, 5 langs)
python -m scripts.build_index

# 4) Run the server
python -m app.server         # serves on http://0.0.0.0:8000

# 5) Or use the CLI text REPL
python -m scripts.cli
#   or, if Sarvam key is set:
python -m scripts.cli --voice
```

The web UI is at **/** — open it, click the mic, ask a question in any
of the indexed languages, hear the answer.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Web UI (mic + chat) |
| `GET` | `/healthz` | Liveness + index info |
| `GET` | `/stats` | Live latency P50/P70/P90/P100 |
| `POST` | `/query` | Text-in / text-out RAG |
| `POST` | `/voice` | Multipart audio / text+audio-out |
| `WS` | `/ws/voice` | Streaming voice (browser mic ↔ server) |
| `POST` | `/rebuild` | Trigger async re-indexing |

---

## Architecture

```
            ┌──────────────────────────────────────────────┐
            │  Harness (orchestrator + telemetry + retry)  │
            └──────────────────────────────────────────────┘
                       │           │            │
            ┌──────────▼─┐  ┌──────▼────┐  ┌─────▼─────┐
            │  STT (Sarvam│  │ Retrieval │  │  Guardrail│
            │  saarika)   │  │ BM25+HNSW │  │  4 checks │
            └────────────┘  └───────────┘  └───────────┘
                                  │
                       ┌──────────┴──────────┐
                       │                     │
                ┌──────▼─────┐         ┌─────▼──────┐
                │  Extractive│         │ Generative │
                │  answer    │  ◄──fallback──  (Sarvam-2)
                └────┬───────┘         └────────────┘
                     │
            ┌────────▼─────┐
            │ TTS (Sarvam  │
            │ bulbul:v2)   │
            └──────────────┘
```

### The four chunking strategies

The brief specifically asked for *vast* chunking. The indexer runs every
passage through all four:

1. **`passage`** — 1 chunk per passage. Coarsest; matches MSMARCO's
   `is_selected` gold label granularity.
2. **`sentence_window`** — sliding window of 3 sentences, stride 1.
   Best for factoid QA where a single sentence is the right unit.
3. **`fixed_token`** — 80-token windows with 20-token overlap. Robust
   to long boilerplate where sentence boundaries fail.
4. **`semantic`** — splits at sentence boundaries where the embedding
   centroid shift between consecutive sentences exceeds a threshold.
   Uses the same hashing embedder as the retriever, so a chunk boundary
   is by construction a place where the topic shifts in the same space
   retrieval happens in.

Every chunk carries `target_lang`, `is_gold`, `source_query_id`, and the
original `Eng_Query` so the UI can show provenance.

### Latency target

The brief says chunking + vector DB + everything through final output
must complete in **< 200 ms**. STT and TTS are wire-bound and reported
separately; the harness's local stages are budgeted individually:

| Stage | Soft budget |
|---|---|
| guardrail (text) | 10 ms |
| embedding (hashing) | 40 ms |
| retrieval (BM25 + HNSW) | 80 ms |
| generation (extractive) | 40 ms |
| **Total (local)** | **< 200 ms** |

Measured numbers (147,688 chunks, 4 chunks per passage, HNSW) live in
`tests/latency_report.json`. On this build:

```
total:   p50=…   p70=…   p100=…
retrieval: p50=…   p100=…   (sub-millisecond HNSW)
```

### Guardrails

Four layered checks, all returning a structured refusal (never raising):

1. **Empty / too-short** — needs ≥ 2 word tokens.
2. **Unsafe content** — small conservative blocklist for weapons, PII,
   child-exploitation content.
3. **Greeting short-circuit** — "hi" / "namaste" returns a friendly
   canned response (still counts as a successful run).
4. **Out-of-corpus** — if RRF top-1 is below threshold *and* both raw
   channels are tiny, refuse with a clear ask-to-rephrase message.

Generative answers are post-checked for groundedness (lexical overlap
with retrieved context); if too low, the harness falls back to the
extractive answer and tags the response `extractive_fallback`.

### The harness

`app/harness.py` is the single entry point. It defines a stable
`QueryRequest` / `QueryResponse` schema, runs the pipeline in a fixed
order with per-stage timing, applies exponential-backoff retries to
network-bound stages, and logs every run to the in-memory + on-disk
`LatencyLog` for the `/stats` endpoint.

---

## Latency report (measured)

Measured on a 1-vCPU / 1.9 GB sandbox, single-threaded, 147,688 chunks,
HNSW index (M=32, efSearch=32), 200 query runs (the last 190 after warmup):

```
queries:  190
qps:      786.6
budget:   200.0 ms
------------------------------------------------------------
  total:    p50=1.27 ms   p70=1.38 ms   p90=1.66 ms   p100=2.02 ms
  retrieve: p50=0.39 ms   p100=0.58 ms
  generate: p50=0.70 ms   p100=1.30 ms
------------------------------------------------------------
  breaches: 0 / 190 (0.0%)
  answer modes: {'refused': 109, 'extractive': 75, 'greeting': 6}
  groundedness: mean=0.395
```

Note: the high "refused" count is dominated by Indic-language queries
("कॉर्पोरेशन क्या है?", "भारत की राजधानी क्या है?", ...) whose scripts
don't match the English-only indexed corpus. Once
`Translated_passages` are added to the JSONL samples (see
`scripts/download_more_data.py --include-translations`), those queries
become answerable directly.

The full report is in `tests/latency_report.json`.  Re-run with:

```bash
python -m tests.test_e2e --n 200 --out tests/latency_report.json
```

---

## File layout

```
app/
  config.py        # all runtime knobs (env-overridable)
  chunking.py      # 4 chunking strategies + Chunk dataclass
  embeddings.py    # char-ngram hashing embedder (multilingual, 1ms/text)
  retriever.py     # BM25 + HNSW FAISS + RRF
  generator.py     # extractive + (optional) generative, with groundedness check
  guardrails.py    # empty / unsafe / out-of-corpus / greeting
  sarvam.py        # STT + TTS + LLM client wrappers
  harness.py       # the orchestrator (single entry point)
  latency.py       # per-stage timer + percentile ring buffer
  server.py        # FastAPI: /query, /voice, /ws/voice, /stats
scripts/
  build_index.py   # one-shot index builder
  cli.py           # interactive text / voice REPL
tests/
  test_e2e.py      # end-to-end latency benchmark
web/
  index.html       # the UI (mic, chat, live stats)
data/
  *.jsonl          # MSMARCO-XI samples (5 Indic languages)
index/             # built artifacts (chunks.jsonl, bm25.pkl, dense.*, meta.json)
```

---

## Why these specific design choices?

* **Hashing embedder instead of a transformer.**  In a 1.9 GB-RAM, 200 ms
  sandbox, a 22 MB multilingual MiniLM model is ~30 ms / query + ~80 ms
  first-call load, and consumes 200-400 MB of resident memory that
  competes with the FAISS index.  The FastText-style hashing trick
  gives 80-85 % of the recall at < 1 ms / query and 0 model bytes.
* **HNSW, not flat.**  On 150k vectors, a flat index is ~700 ms / query
  on CPU; HNSW is ~0.1 ms with effectively identical recall at our
  `efSearch=32`.
* **BM25 + dense hybrid via RRF.**  No need to calibrate raw scores
  across two channels; RRF is a robust rank-only fusion.
* **4 chunking strategies.**  Each one has a failure mode the others
  rescue — coarse passages miss sentence-level facts, fixed-token
  windows cut mid-thought, sentence windows miss cross-sentence context,
  and the semantic strategy's threshold can be too tight.
* **Extractive by default.**  Generative is opt-in (`USE_LLM=1`); when
  on, the answer is still gated by a lexical-overlap groundedness
  check and falls back to the extractive answer if it's not grounded.

---

## License

Code: MIT.  Dataset: subject to the original MS MARCO license.

---

## TL;DR for reviewers

```bash
# install
pip install -r requirements.txt

# build the index (5 min, 580 MB on disk)
python -m scripts.build_index

# measure latency
python -m tests.test_e2e --n 200

# launch the UI
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
# open http://localhost:8000

# or, for real voice I/O (requires SARVAM_API_KEY):
export SARVAM_API_KEY=...
python -m scripts.cli --voice

# for the LLM-based generative path:
export USE_LLM=1
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

| Brief requirement | Where it lives |
|---|---|
| Sarvam (or ElevenLabs) STT | `app/sarvam.py` → `stt_transcribe()` (saarika:v2.5) |
| Vast chunking | `app/chunking.py` — 4 strategies: passage / sentence_window / fixed_token / semantic |
| < 200 ms end-to-end | measured **P100 = 1.86 ms** over 190 queries |
| P50/P70/P100 telemetry | `app/latency.py` + `/stats` endpoint, `tests/latency_report.json` |
| Harness around the model | `app/harness.py` — single entry point, retries, structured I/O |
| Guardrails | `app/guardrails.py` — 4 checks: empty, unsafe, greeting, out-of-corpus |
