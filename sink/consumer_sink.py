"""Persistence sink: consumes transactions.scored and transactions.dlq and
writes them to PostgreSQL.

Delivery: Kafka offsets are committed only AFTER the database transaction
commits, so rows are durable before their offsets advance. Scored rows use
ON CONFLICT DO NOTHING on the (transaction_id, processing_ts) primary key, so a
scored message redelivered after a crash between the database commit and the
offset commit is ignored.

Pure batching/mapping logic lives in module-level functions and the Batcher
class (unit-tested without a database); Kafka/PostgreSQL wiring is in main().

Env: KAFKA_BOOTSTRAP, TOPIC_SCORED, TOPIC_DLQ, DATABASE_URL,
BATCH_ROWS (500), BATCH_WAIT_S (2.0).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC_SCORED = os.environ.get("TOPIC_SCORED", "transactions.scored")
TOPIC_DLQ = os.environ.get("TOPIC_DLQ", "transactions.dlq")
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://fraud:fraud@postgres:5432/fraud")
BATCH_ROWS = int(os.environ.get("BATCH_ROWS", "500"))
BATCH_WAIT_S = float(os.environ.get("BATCH_WAIT_S", "2.0"))

SCORED_SQL = """
INSERT INTO scored_transactions
  (transaction_id, processing_ts, event_time, fraud_probability, is_fraud_pred,
   decision, model_version, latency_ms, error, warnings)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (transaction_id, processing_ts) DO NOTHING
"""
DLQ_SQL = "INSERT INTO dlq_events (error, raw) VALUES (%s, %s)"


def map_scored(payload: bytes | str) -> Optional[tuple]:
    """Scored-topic JSON -> row tuple for SCORED_SQL; None if unusable."""
    try:
        ev = json.loads(payload)
    except Exception:
        return None
    if not isinstance(ev, dict):
        return None
    ts = ev.get("processing_ts")
    if not isinstance(ts, (int, float)):
        ts = time.time()
    warnings = ev.get("warnings")
    return (
        str(ev.get("transaction_id")),
        float(ts),
        datetime.fromtimestamp(float(ts), tz=timezone.utc),
        ev.get("fraud_probability"),
        ev.get("is_fraud_pred"),
        str(ev.get("decision")),
        ev.get("model_version"),
        ev.get("latency_ms"),
        ev.get("error"),
        json.dumps(warnings) if warnings is not None else None,
    )


def map_dlq(payload: bytes | str) -> tuple:
    """DLQ-topic JSON -> row tuple for DLQ_SQL (never rejects: the DLQ is the
    place of last resort, so even unparseable payloads are stored raw)."""
    raw = payload.decode("utf-8", "replace") if isinstance(payload, bytes) else payload
    try:
        rec = json.loads(raw)
        if isinstance(rec, dict):
            return (rec.get("error"), rec.get("raw"))
    except Exception:
        pass
    return ("unparseable_dlq_record", raw[:65536])


class Batcher:
    """Size- and time-bounded batch accumulator (pure; no I/O)."""

    def __init__(self, max_rows: int = 500, max_wait_s: float = 2.0):
        self.max_rows = max_rows
        self.max_wait_s = max_wait_s
        self.rows: list[tuple] = []
        self._first_ts: Optional[float] = None

    def add(self, row: tuple, now: Optional[float] = None) -> bool:
        """Append a row; returns True when the batch should be flushed."""
        if self._first_ts is None:
            self._first_ts = now if now is not None else time.time()
        self.rows.append(row)
        return self.should_flush(now)

    def should_flush(self, now: Optional[float] = None) -> bool:
        if not self.rows:
            return False
        if len(self.rows) >= self.max_rows:
            return True
        now = now if now is not None else time.time()
        return (now - self._first_ts) >= self.max_wait_s

    def drain(self) -> list[tuple]:
        rows, self.rows, self._first_ts = self.rows, [], None
        return rows


def main() -> None:  # pragma: no cover - integration wiring
    import psycopg
    from confluent_kafka import Consumer

    conn = None
    while conn is None:  # wait for postgres readiness on cold start
        try:
            conn = psycopg.connect(DATABASE_URL)
        except Exception as e:
            print(f"[sink] waiting for database: {e}", flush=True)
            time.sleep(3)
    conn.autocommit = False
    print("[sink] connected to database", flush=True)

    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": "scored-sink",
        "auto.offset.reset": "earliest",   # persist everything from the start
        "enable.auto.commit": False,       # commit only after DB commit
    })
    consumer.subscribe([TOPIC_SCORED, TOPIC_DLQ])

    scored = Batcher(BATCH_ROWS, BATCH_WAIT_S)
    dlq = Batcher(BATCH_ROWS, BATCH_WAIT_S)
    total_scored = total_dlq = 0

    def flush() -> None:
        nonlocal total_scored, total_dlq
        s_rows, d_rows = scored.drain(), dlq.drain()
        if not s_rows and not d_rows:
            return
        with conn.cursor() as cur:
            if s_rows:
                cur.executemany(SCORED_SQL, s_rows)
            if d_rows:
                cur.executemany(DLQ_SQL, d_rows)
        conn.commit()          # 1) durable in the database
        consumer.commit(asynchronous=False)  # 2) then advance offsets
        total_scored += len(s_rows)
        total_dlq += len(d_rows)
        print(f"[sink] flushed scored={len(s_rows)} dlq={len(d_rows)} "
              f"(totals {total_scored}/{total_dlq})", flush=True)

    while True:
        msg = consumer.poll(0.5)
        if msg is not None and msg.error() is None:
            if msg.topic() == TOPIC_SCORED:
                row = map_scored(msg.value())
                if row is not None:
                    scored.add(row)
            elif msg.topic() == TOPIC_DLQ:
                dlq.add(map_dlq(msg.value()))
        if scored.should_flush() or dlq.should_flush():
            try:
                flush()
            except Exception as e:
                conn.rollback()
                print(f"[sink] flush failed, will retry: {e}", flush=True)
                time.sleep(2)


if __name__ == "__main__":
    main()
