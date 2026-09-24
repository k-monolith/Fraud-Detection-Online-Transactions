"""Drift computation for the metrics exporter — pure logic,
no Kafka/Prometheus imports, unit-testable.

PSI (Population Stability Index) between a reference distribution captured at
training time and a rolling live window. Conventional reading: <0.1 stable,
0.1-0.2 moderate shift, >0.2 significant shift (alert threshold).
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from typing import Any, Optional

EPS = 1e-6

# Drift feature set: top-10 by importance from features.json, EXCLUDING
# TransactionDT (a monotone clock -- PSI on it is always "drifting" by
# construction, especially under replay of future data) and taking the next
# ranked feature (addr1) in its place. card6 is categorical.
DRIFT_FEATURES = ["C13", "TransactionAmt", "card1", "C1", "V258",
                  "card6", "card2", "V70", "C14", "addr1"]
CATEGORICAL_DRIFT = {"card6"}
SCORE_BINS = [i / 20.0 for i in range(21)]  # 20 equal-width bins on [0,1]


def psi(ref_props: list[float], act_props: list[float]) -> float:
    """Standard PSI; both inputs are same-length probability vectors."""
    s = 0.0
    for r, a in zip(ref_props, act_props):
        r = max(r, EPS)
        a = max(a, EPS)
        s += (a - r) * math.log(a / r)
    return s


def numeric_props(values: list[float], edges: list[float]) -> list[float]:
    """Histogram proportions over fixed edges; values outside clamp to end bins."""
    n_bins = len(edges) - 1
    counts = [0] * n_bins
    total = 0
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        i = n_bins - 1
        for b in range(n_bins):
            if v < edges[b + 1]:
                i = b
                break
        counts[i] += 1
        total += 1
    if total == 0:
        return [0.0] * n_bins
    return [c / total for c in counts]


def categorical_props(values: list, categories: list[str]) -> list[float]:
    """Proportions over reference categories + a trailing OTHER bucket."""
    idx = {c: i for i, c in enumerate(categories)}
    counts = [0] * (len(categories) + 1)
    total = 0
    for v in values:
        if v is None:
            continue
        counts[idx.get(str(v), len(categories))] += 1
        total += 1
    if total == 0:
        return [0.0] * len(counts)
    return [c / total for c in counts]


def quantile_edges(values: list[float], n_bins: int = 10) -> list[float]:
    """Quantile bin edges from a sample (dedup-safe, min 2 bins)."""
    xs = sorted(v for v in values
                if v is not None and not (isinstance(v, float) and math.isnan(v)))
    if len(xs) < 2:
        return [0.0, 1.0]
    edges = []
    for q in range(n_bins + 1):
        edges.append(xs[min(len(xs) - 1, int(q * (len(xs) - 1) / n_bins))])
    out = [edges[0] - abs(edges[0]) * 0.001 - 1e-9]
    for e in edges[1:]:
        if e > out[-1]:
            out.append(e)
    out[-1] = out[-1] + abs(out[-1]) * 0.001 + 1e-9
    return out if len(out) >= 3 else [out[0], (out[0] + out[-1]) / 2, out[-1]]


class RollingWindow:
    """Time-bounded sample store: score, decision, latency, drift features."""

    def __init__(self, max_age_s: float = 300.0, max_items: int = 100_000):
        self.max_age_s = max_age_s
        self.items: deque = deque(maxlen=max_items)

    def add(self, sample: dict, ts: Optional[float] = None) -> None:
        self.items.append((ts if ts is not None else time.time(), sample))

    def _evict(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        while self.items and now - self.items[0][0] > self.max_age_s:
            self.items.popleft()

    def snapshot(self, key: str, now: Optional[float] = None) -> list:
        self._evict(now)
        return [s[key] for _, s in self.items if key in s and s[key] is not None]

    def size(self, now: Optional[float] = None) -> int:
        self._evict(now)
        return len(self.items)


class DriftTracker:
    """Holds the reference (loaded or bootstrapped) and computes PSI gauges."""

    def __init__(self, reference: Optional[dict] = None,
                 bootstrap_n: int = 5000):
        self.reference = reference
        self.bootstrap_n = bootstrap_n
        self.bootstrap_used = reference is None
        self._boot_scores: list[float] = []
        self._boot_feats: dict[str, list] = {f: [] for f in DRIFT_FEATURES}

    @classmethod
    def from_file(cls, path: str) -> "DriftTracker":
        with open(path) as f:
            return cls(reference=json.load(f))

    def observe_for_bootstrap(self, score: Optional[float],
                              feats: dict[str, Any]) -> None:
        """Accumulate early traffic into a self-reference when no file exists."""
        if self.reference is not None:
            return
        if score is not None:
            self._boot_scores.append(score)
        for f in DRIFT_FEATURES:
            if f in feats and feats[f] is not None                     and len(self._boot_feats[f]) < self.bootstrap_n:
                self._boot_feats[f].append(feats[f])
        if len(self._boot_scores) >= self.bootstrap_n:
            self._freeze_bootstrap()

    def _freeze_bootstrap(self) -> None:
        ref: dict = {"source": "bootstrap",
                     "score": {"edges": SCORE_BINS,
                               "props": numeric_props(self._boot_scores,
                                                      SCORE_BINS)},
                     "features": {}}
        for f in DRIFT_FEATURES:
            vals = self._boot_feats[f]
            if f in CATEGORICAL_DRIFT:
                seen: dict[str, int] = {}
                for v in vals:
                    seen[str(v)] = seen.get(str(v), 0) + 1
                cats = [c for c, _ in sorted(seen.items(),
                                             key=lambda kv: -kv[1])[:20]]
                ref["features"][f] = {
                    "type": "categorical", "categories": cats,
                    "props": categorical_props(vals, cats)}
            else:
                edges = quantile_edges([float(v) for v in vals
                                        if _is_num(v)], 10)
                ref["features"][f] = {
                    "type": "numeric", "edges": edges,
                    "props": numeric_props([float(v) for v in vals
                                            if _is_num(v)], edges)}
        self.reference = ref
        self._boot_scores.clear()
        self._boot_feats = {f: [] for f in DRIFT_FEATURES}

    def ready(self) -> bool:
        return self.reference is not None

    def score_psi(self, live_scores: list[float]) -> Optional[float]:
        if not self.ready() or len(live_scores) < 100:
            return None
        ref = self.reference["score"]
        return psi(ref["props"], numeric_props(live_scores, ref["edges"]))

    def feature_psi(self, name: str, live_values: list) -> Optional[float]:
        if not self.ready() or name not in self.reference["features"]:
            return None
        if len([v for v in live_values if v is not None]) < 100:
            return None
        ref = self.reference["features"][name]
        if ref["type"] == "categorical":
            return psi(ref["props"],
                       categorical_props(live_values, ref["categories"]))
        vals = [float(v) for v in live_values if _is_num(v)]
        return psi(ref["props"], numeric_props(vals, ref["edges"]))


def _is_num(v) -> bool:
    if v is None:
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return not math.isnan(f)
