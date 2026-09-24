"""Model-behavior metrics exporter.

Consumes transactions.raw (feature drift), transactions.scored (score
distribution, flag rate, fallback rate, per-event latency) and transactions.dlq
(dead-letter rate); exposes Prometheus metrics on :8000/metrics.

Reference distributions for PSI: artifacts/reference_distributions.json when
present (built once by tools/make_reference.py from the training CSVs);
otherwise built from the first BOOTSTRAP_N live events and reported by
reference_bootstrap_used=1.

Env: KAFKA_BOOTSTRAP, TOPIC_RAW, TOPIC_SCORED, TOPIC_DLQ,
ARTIFACTS_DIR, EXPORTER_PORT (8000), WINDOW_S (300), BOOTSTRAP_N (5000),
SCORING_GROUP (fraud-scoring, for consumer-lag polling).
"""
from __future__ import annotations

import json
import os
import threading
import time

from confluent_kafka import Consumer, ConsumerGroupTopicPartitions, TopicPartition
from confluent_kafka.admin import AdminClient
from prometheus_client import Counter, Gauge, Histogram, start_http_server

import drift
from drift import DriftTracker, RollingWindow, DRIFT_FEATURES

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC_RAW = os.environ.get("TOPIC_RAW", "transactions.raw")
TOPIC_SCORED = os.environ.get("TOPIC_SCORED", "transactions.scored")
TOPIC_DLQ = os.environ.get("TOPIC_DLQ", "transactions.dlq")
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/opt/artifacts")
PORT = int(os.environ.get("EXPORTER_PORT", "8000"))
WINDOW_S = float(os.environ.get("WINDOW_S", "300"))
BOOTSTRAP_N = int(os.environ.get("BOOTSTRAP_N", "5000"))
SCORING_GROUP = os.environ.get("SCORING_GROUP", "fraud-scoring")
REFERENCE_PATH = os.path.join(ARTIFACTS_DIR, "reference_distributions.json")

# ---- Prometheus metrics ------------------------------------------------------
RAW_TOTAL = Counter("raw_events_total", "Raw transaction events observed")
SCORED_TOTAL = Counter("scored_events_total", "Scored events", ["decision"])
DLQ_TOTAL = Counter("dlq_events_total", "Dead-letter events")
SCORE_HIST = Histogram("fraud_score", "Fraud probability distribution",
                       buckets=[i / 20 for i in range(1, 21)])
LATENCY_HIST = Histogram("scoring_latency_ms", "Per-event scoring latency (ms)",
                         buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
FLAG_RATE = Gauge("fraud_flag_rate", "Share of is_fraud_pred=1 in the window")
FALLBACK_RATE = Gauge("fallback_rate",
                      "Share of decision=REVIEW (fail-open) in the window")
SCORE_PSI = Gauge("score_psi", "PSI of live scores vs training reference")
FEATURE_PSI = Gauge("feature_psi", "PSI of live feature vs training reference",
                    ["feature"])
CONSUMER_LAG = Gauge("scoring_consumer_lag",
                     "Lag of the scoring job on transactions.raw", ["partition"])
BOOT_USED = Gauge("reference_bootstrap_used",
                  "1 if PSI reference was bootstrapped from live traffic")
WINDOW_SIZE = Gauge("window_events", "Events currently in the rolling window")


def load_tracker() -> DriftTracker:
    if os.path.exists(REFERENCE_PATH):
        print(f"[exporter] PSI reference loaded from {REFERENCE_PATH}", flush=True)
        t = DriftTracker.from_file(REFERENCE_PATH)
        BOOT_USED.set(0)
        return t
    print(f"[exporter] no reference file at {REFERENCE_PATH}; "
          f"bootstrapping from first {BOOTSTRAP_N} events", flush=True)
    BOOT_USED.set(1)
    return DriftTracker(reference=None, bootstrap_n=BOOTSTRAP_N)


class Exporter:
    def __init__(self):
        self.window = RollingWindow(max_age_s=WINDOW_S)
        self.tracker = load_tracker()
        self.lock = threading.Lock()

    # ---- event handlers ------------------------------------------------------
    def on_raw(self, payload: bytes) -> None:
        RAW_TOTAL.inc()
        try:
            ev = json.loads(payload)
        except Exception:
            return
        if not isinstance(ev, dict):
            return
        feats = {f: ev.get(f) for f in DRIFT_FEATURES}
        with self.lock:
            self.window.add({"feats": feats, **{f"f:{k}": v
                                                for k, v in feats.items()}})
            # Feature reference is bootstrapped straight from raw traffic: the
            # raw stream runs far ahead of scored output (scoring is slower
            # than the producer), so a transaction-id join would never match.
            if not self.tracker.ready():
                self.tracker.observe_for_bootstrap(None, feats)

    def on_scored(self, payload: bytes) -> None:
        try:
            ev = json.loads(payload)
        except Exception:
            return
        decision = str(ev.get("decision"))
        SCORED_TOTAL.labels(decision=decision).inc()
        score = ev.get("fraud_probability")
        lat = ev.get("latency_ms")
        if isinstance(score, (int, float)):
            SCORE_HIST.observe(float(score))
        if isinstance(lat, (int, float)):
            LATENCY_HIST.observe(float(lat))
        with self.lock:
            self.window.add({
                "score": float(score) if isinstance(score, (int, float)) else None,
                "flag": ev.get("is_fraud_pred"),
                "review": 1 if decision == "REVIEW" else 0,
            })
            if not self.tracker.ready():
                self.tracker.observe_for_bootstrap(
                    float(score) if isinstance(score, (int, float)) else None,
                    {})
                if self.tracker.ready():
                    print("[exporter] bootstrap reference frozen", flush=True)

    def on_dlq(self, payload: bytes) -> None:
        DLQ_TOTAL.inc()

    # ---- periodic gauge computation -----------------------------------------
    def recompute(self) -> None:
        with self.lock:
            flags = self.window.snapshot("flag")
            reviews = self.window.snapshot("review")
            scores = self.window.snapshot("score")
            WINDOW_SIZE.set(self.window.size())
            if flags:
                FLAG_RATE.set(sum(1 for f in flags if f == 1) / len(flags))
            if reviews:
                FALLBACK_RATE.set(sum(reviews) / len(reviews))
            sp = self.tracker.score_psi(scores)
            if sp is not None:
                SCORE_PSI.set(sp)
            for f in DRIFT_FEATURES:
                vals = self.window.snapshot(f"f:{f}")
                fp = self.tracker.feature_psi(f, vals)
                if fp is not None:
                    FEATURE_PSI.labels(feature=f).set(fp)

    def poll_lag(self, admin: AdminClient, consumer: Consumer) -> None:
        """Scoring-job lag on transactions.raw."""
        try:
            md = admin.list_topics(TOPIC_RAW, timeout=5)
            parts = md.topics[TOPIC_RAW].partitions
            tps = [TopicPartition(TOPIC_RAW, p) for p in parts]
            # confluent-kafka >=2.x: takes [ConsumerGroupTopicPartitions],
            # returns {group: future}
            fut = admin.list_consumer_group_offsets(
                [ConsumerGroupTopicPartitions(SCORING_GROUP, tps)])
            res = list(fut.values())[0].result(timeout=5)
            committed = {tp.partition: tp.offset
                         for tp in res.topic_partitions
                         if tp.topic == TOPIC_RAW}
            for tp in tps:
                _, end = consumer.get_watermark_offsets(tp, timeout=5)
                off = committed.get(tp.partition)
                if off is not None and off >= 0:
                    CONSUMER_LAG.labels(partition=str(tp.partition)) \
                                .set(max(end - off, 0))
        except Exception as e:  # lag is best-effort; never kill the exporter
            print(f"[exporter] lag poll failed: {e}", flush=True)


def main() -> None:
    start_http_server(PORT)
    print(f"[exporter] /metrics on :{PORT}", flush=True)
    exp = Exporter()
    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": "metrics-exporter",
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([TOPIC_RAW, TOPIC_SCORED, TOPIC_DLQ])
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP})

    handlers = {TOPIC_RAW: exp.on_raw, TOPIC_SCORED: exp.on_scored,
                TOPIC_DLQ: exp.on_dlq}
    last_recompute = last_lag = 0.0
    while True:
        msg = consumer.poll(0.5)
        now = time.time()
        if msg is not None and msg.error() is None:
            handlers.get(msg.topic(), lambda p: None)(msg.value())
        if now - last_recompute >= 15:
            exp.recompute()
            last_recompute = now
        if now - last_lag >= 15:
            exp.poll_lag(admin, consumer)
            last_lag = now


if __name__ == "__main__":
    main()
