"""Latency telemetry.

Every pipeline run records per-stage timings.  We also keep a rolling
in-memory log so the server can report live P50/P70/P100 without disk
I/O, plus a JSONL log on disk for offline analysis.
"""
from __future__ import annotations

import bisect
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque

from .config import INDEX_DIR


@dataclass
class StageTimer:
    """One run's timings.  All values in milliseconds."""
    stages: dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    over_budget: bool = False
    budget: float = 200.0

    def add(self, name: str, ms: float):
        self.stages[name] = round(ms, 3)

    def finish(self, t0: float, budget: float = 200.0):
        self.total = round((time.perf_counter() - t0) * 1000, 3)
        self.over_budget = self.total > budget
        self.budget = budget

    def to_dict(self) -> dict:
        return {"stages": self.stages, "total_ms": self.total,
                "over_budget": self.over_budget, "budget_ms": self.budget}


# ---------- ring buffer with percentile query ----------
class LatencyLog:
    """Thread-safe ring of recent total latencies + per-stage."""
    def __init__(self, capacity: int = 5000, log_path: Path | None = None):
        self.capacity = capacity
        self.totals: Deque[float] = deque(maxlen=capacity)
        self.stages: dict[str, Deque[float]] = {}
        self.lock = threading.Lock()
        self.log_path = log_path or (INDEX_DIR / "latency.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, t: StageTimer):
        with self.lock:
            self.totals.append(t.total)
            for k, v in t.stages.items():
                self.stages.setdefault(k, deque(maxlen=self.capacity)).append(v)
        # also append to disk (best effort, non-blocking-ish)
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def stats(self) -> dict:
        with self.lock:
            totals = list(self.totals)
            stages = {k: list(v) for k, v in self.stages.items()}
        if not totals:
            return {"n": 0}

        def pct(xs, p):
            if not xs:
                return 0.0
            xs = sorted(xs)
            i = max(0, min(len(xs) - 1, int(math.ceil(p / 100.0 * len(xs))) - 1))
            return xs[i]

        return {
            "n": len(totals),
            "p50_ms": round(pct(totals, 50), 3),
            "p70_ms": round(pct(totals, 70), 3),
            "p90_ms": round(pct(totals, 90), 3),
            "p100_ms": round(pct(totals, 100), 3),
            "mean_ms": round(sum(totals) / len(totals), 3),
            "max_ms": round(max(totals), 3),
            "min_ms": round(min(totals), 3),
            "budget_breaches": sum(1 for x in totals if x > 200.0),
            "stages": {k: {
                "n": len(v),
                "p50_ms": round(pct(v, 50), 3),
                "p90_ms": round(pct(v, 90), 3),
                "p100_ms": round(pct(v, 100), 3),
                "mean_ms": round(sum(v) / len(v), 3) if v else 0.0,
            } for k, v in stages.items()},
        }


# Singleton -- one log per process.
LOG = LatencyLog()
