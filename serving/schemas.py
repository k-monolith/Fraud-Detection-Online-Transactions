"""Event schema, normalization and validation.

Pure functions only — no Flink, no model imports — so this module is used by
the Flink job and unit-tested on its own.

Data contract (mirrors training, manifest.json "data_contract"):
- One JSON object per event, keys = raw dataset columns (TransactionID included).
- Identity columns may arrive hyphenated ("id-01", competition test-file quirk) —
  normalized here to underscore form ("id_01").
- Missing fields are tolerated (imputed downstream by the pickled pipeline).
- Non-null categorical values are coerced to str; numerics to float.
- Anything that is not a JSON object with a usable TransactionID is malformed
  and belongs on the DLQ — the job must never crash on input.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

_ID_HYPHEN = re.compile(r"^id-(\d{2})$")


def normalize_keys(event: dict) -> dict:
    """Rename 'id-XX' keys to 'id_XX' and strip surrounding whitespace from keys."""
    out = {}
    for k, v in event.items():
        k = str(k).strip()
        m = _ID_HYPHEN.match(k)
        if m:
            k = f"id_{m.group(1)}"
        out[k] = v
    return out


def parse_event(raw: str | bytes) -> tuple[Optional[dict], Optional[str]]:
    """Parse a Kafka record into a JSON object. Returns (event, error)."""
    try:
        obj = json.loads(raw)
    except Exception as e:  # malformed JSON -> DLQ
        return None, f"json_parse_error: {e}"
    if not isinstance(obj, dict) or not obj:
        return None, "not_a_json_object"
    return normalize_keys(obj), None


def _to_float(v: Any) -> tuple[Optional[float], bool]:
    """Coerce to finite float; None/'' -> None. Returns (value, ok)."""
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None, True
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None, False
    if math.isnan(f) or math.isinf(f):
        return None, True
    return f, True


def _to_str(v: Any) -> Optional[str]:
    """Coerce non-null categorical values to str (training data contract)."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s if s != "" else None


def coerce_and_complete(
    event: dict,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[dict, list[str]]:
    """Build a complete raw row for the preprocessing pipeline.

    Every expected column is present in the result (missing -> None); values are
    type-coerced per the data contract. Unknown keys are ignored (forward
    compatibility). Returns (row, warnings). Warnings never reject an event —
    rejection is reserved for structurally malformed input (parse_event /
    extract_transaction_id).
    """
    warnings: list[str] = []
    row: dict[str, Any] = {}
    for c in numeric_cols:
        val, ok = _to_float(event.get(c))
        if not ok:
            warnings.append(f"uncoercible_numeric:{c}")
        row[c] = val
    for c in categorical_cols:
        row[c] = _to_str(event.get(c))
    known = set(numeric_cols) | set(categorical_cols) | {"TransactionID"}
    unknown = [k for k in event if k not in known]
    if unknown:
        warnings.append(f"ignored_unknown_fields:{len(unknown)}")
    return row, warnings


def extract_transaction_id(event: dict) -> Optional[str]:
    """Event identifier carried through to the scored topic."""
    tid = event.get("TransactionID")
    if tid is None:
        return None
    s = str(tid).strip()
    return s or None


def make_dlq_record(raw: str | bytes, error: str) -> str:
    """Envelope for the dead-letter topic (original payload preserved)."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return json.dumps({"error": error, "raw": raw[:65536]})
