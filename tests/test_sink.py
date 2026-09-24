"""Unit tests for sink/consumer_sink.py — mapping and batching only, no DB."""
import json
import sys, os
from datetime import timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sink"))

from consumer_sink import Batcher, map_scored, map_dlq


def test_map_scored_full_record():
    payload = json.dumps({
        "transaction_id": "3000001", "fraud_probability": 0.73,
        "is_fraud_pred": 1, "decision": "ALERT",
        "model_version": "20260921T144658Z", "processing_ts": 1790000000.5,
        "latency_ms": 21.4, "error": None, "warnings": ["ignored_unknown_fields:2"],
    })
    row = map_scored(payload)
    assert row[0] == "3000001" and row[1] == 1790000000.5
    assert row[2].tzinfo == timezone.utc          # event_time is tz-aware UTC
    assert row[3] == 0.73 and row[4] == 1 and row[5] == "ALERT"
    assert json.loads(row[9]) == ["ignored_unknown_fields:2"]


def test_map_scored_review_record_with_nulls():
    row = map_scored(json.dumps({
        "transaction_id": "9", "fraud_probability": None, "is_fraud_pred": None,
        "decision": "REVIEW", "processing_ts": 1790000001.0,
        "error": "RuntimeError: boom", "warnings": None,
    }))
    assert row[3] is None and row[4] is None and row[5] == "REVIEW"
    assert row[8] == "RuntimeError: boom" and row[9] is None


def test_map_scored_rejects_garbage():
    assert map_scored(b"{{{not json") is None
    assert map_scored(json.dumps([1, 2, 3])) is None


def test_map_dlq_wraps_anything():
    err, raw = map_dlq(json.dumps({"error": "json_parse_error: x", "raw": "{bro"}))
    assert err == "json_parse_error: x" and raw == "{bro"
    err2, raw2 = map_dlq(b"\xff\xfe not even json")
    assert err2 == "unparseable_dlq_record" and "not even json" in raw2


def test_batcher_flushes_on_size():
    b = Batcher(max_rows=3, max_wait_s=999)
    assert b.add(("r1",), now=100.0) is False
    assert b.add(("r2",), now=100.1) is False
    assert b.add(("r3",), now=100.2) is True
    assert b.drain() == [("r1",), ("r2",), ("r3",)]
    assert b.rows == [] and b.should_flush(now=200.0) is False


def test_batcher_flushes_on_age():
    b = Batcher(max_rows=1000, max_wait_s=2.0)
    b.add(("r1",), now=100.0)
    assert b.should_flush(now=101.0) is False
    assert b.should_flush(now=102.5) is True
    b.drain()
    b.add(("r2",), now=200.0)                    # age resets after drain
    assert b.should_flush(now=201.0) is False
