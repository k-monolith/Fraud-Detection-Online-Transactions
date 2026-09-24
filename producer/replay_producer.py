"""Replay producer: streams the unlabeled IEEE-CIS test rows to
transactions.raw as JSON events at a configurable rate, simulating live
traffic. No labels are sent, as in production.

Env configuration:
  KAFKA_BOOTSTRAP   default kafka:9092
  TOPIC_RAW         default transactions.raw
  DATA_DIR          default /data (expects test_transaction.csv, test_identity.csv)
  RATE_EPS          default 50      events per second
  BURST_EVERY_S     default 0       every N seconds, send a burst (0 = off)
  BURST_SIZE        default 500     events per burst
  LOOP              default true    restart from the beginning when exhausted
  MALFORMED_PPM     default 200     inject ~N malformed events per million (DLQ demo)
"""
from __future__ import annotations

import itertools
import json
import math
import os
import random
import sys
import time

import pandas as pd
from confluent_kafka import Producer

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC = os.environ.get("TOPIC_RAW", "transactions.raw")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
RATE = float(os.environ.get("RATE_EPS", "50"))
BURST_EVERY = float(os.environ.get("BURST_EVERY_S", "0"))
BURST_SIZE = int(os.environ.get("BURST_SIZE", "500"))
LOOP = os.environ.get("LOOP", "true").lower() == "true"
MALFORMED_PPM = int(os.environ.get("MALFORMED_PPM", "200"))
CHUNK = 20_000

random.seed(42)


def normalize_id_cols(cols: list[str]) -> dict:
    """Competition quirk: test_identity uses 'id-01' where train uses 'id_01'."""
    return {c: c.replace("id-", "id_") if c.startswith("id-") else c for c in cols}


def load_identity(path: str) -> dict[int, dict]:
    idn = pd.read_csv(path)
    idn = idn.rename(columns=normalize_id_cols(list(idn.columns)))
    idn = idn.set_index("TransactionID")
    print(f"[producer] identity rows: {len(idn):,}", flush=True)
    return {int(k): {c: v for c, v in rec.items() if not _is_nan(v)}
            for k, rec in idn.to_dict(orient="index").items()}


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def row_to_event(row: dict, identity: dict) -> dict:
    event = {k: (None if _is_nan(v) else v) for k, v in row.items()}
    tid = event.get("TransactionID")
    if tid is not None and int(tid) in identity:
        event.update(identity[int(tid)])
    return event


def iter_events(identity: dict):
    path = os.path.join(DATA_DIR, "test_transaction.csv")
    for chunk in pd.read_csv(path, chunksize=CHUNK):
        for row in chunk.to_dict(orient="records"):
            yield row_to_event(row, identity)


def maybe_malform(payload: str, counter: int) -> str:
    if MALFORMED_PPM and random.random() < MALFORMED_PPM / 1_000_000:
        return payload[: max(1, len(payload) // 3)]  # truncated JSON -> DLQ
    return payload


def main() -> None:
    producer = Producer({
        "bootstrap.servers": BOOTSTRAP,
        "linger.ms": 20,
        "batch.num.messages": 1000,
        "compression.type": "lz4",
    })
    identity = load_identity(os.path.join(DATA_DIR, "test_identity.csv"))

    interval = 1.0 / RATE if RATE > 0 else 0.0
    sent, t_start, next_burst = 0, time.time(), (time.time() + BURST_EVERY
                                                 if BURST_EVERY else None)
    rounds = itertools.count() if LOOP else range(1)
    for _ in rounds:
        for event in iter_events(identity):
            payload = maybe_malform(
                json.dumps(event, allow_nan=False, default=str), sent
            )
            producer.produce(TOPIC, value=payload.encode("utf-8"))
            sent += 1
            if sent % 1000 == 0:
                producer.poll(0)
                eps = sent / max(time.time() - t_start, 1e-9)
                print(f"[producer] sent={sent:,} avg_eps={eps:.1f}", flush=True)
            if next_burst and time.time() >= next_burst:
                for burst_event in itertools.islice(iter_events(identity), BURST_SIZE):
                    producer.produce(
                        TOPIC,
                        value=json.dumps(
                            burst_event, allow_nan=False, default=str
                        ).encode("utf-8"),
                    )
                sent += BURST_SIZE
                next_burst = time.time() + BURST_EVERY
                print(f"[producer] burst of {BURST_SIZE} sent", flush=True)
            if interval:
                time.sleep(interval)
        producer.flush()
        if not LOOP:
            break
    print(f"[producer] done, total sent={sent:,}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
