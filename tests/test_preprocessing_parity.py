"""Scoring consistency test.

Runs N synthetic raw rows through (a) the direct path — the pickled pipeline
and models called directly — and (b) the serving path — ScoringService.score()
on the JSON round-tripped event, as the Flink operator does — and asserts
identical fraud probabilities to 1e-6.

Synthetic rows deliberately exercise the hard paths: missing fields, NaN,
unseen categories (reserved -1 code), string-typed numerics, hyphenated
identity keys.

Requires the artifacts bundle; skipped if ARTIFACTS_DIR (default
../artifacts) is absent.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "serving"))

ARTIFACTS_DIR = os.environ.get(
    "ARTIFACTS_DIR",
    os.path.join(os.path.dirname(__file__), "..", "artifacts"),
)
N_ROWS = int(os.environ.get("PARITY_ROWS", "50"))
TOL = 1e-6

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ARTIFACTS_DIR, "manifest.json")),
    reason=f"artifacts bundle not found at {ARTIFACTS_DIR}",
)


@pytest.fixture(scope="module")
def service():
    from scoring_udf import ScoringService

    return ScoringService(ARTIFACTS_DIR)


def _synthetic_events(service, n: int) -> list[dict]:
    rng = np.random.RandomState(42)
    events = []
    for i in range(n):
        ev = {"TransactionID": 3_000_000 + i}
        for c in service.numeric_cols:
            r = rng.rand()
            if r < 0.30:
                continue                      # missing field
            elif r < 0.35:
                ev[c] = None                  # explicit null
            elif r < 0.40:
                ev[c] = str(round(rng.rand() * 100, 2))  # string-typed numeric
            else:
                ev[c] = float(np.round(rng.rand() * 1000, 4))
        for c in service.categorical_cols:
            r = rng.rand()
            if r < 0.40:
                continue
            elif r < 0.55:
                ev[c] = f"unseen_cat_{rng.randint(1e6)}"  # forces -1 code
            else:
                ev[c] = "visa" if rng.rand() < 0.5 else "T"
        if i % 7 == 0 and "id_01" in service.numeric_cols:
            ev["id-01"] = -5.0                # hyphenated key must normalize
            ev.pop("id_01", None)
        events.append(ev)
    return events


def _direct_proba(service, event: dict) -> float:
    """Direct path: pipeline.transform + models, without the serving wrapper."""
    import schemas

    ev = schemas.normalize_keys(event)
    row, _ = schemas.coerce_and_complete(
        ev, service.numeric_cols, service.categorical_cols
    )
    X = pd.DataFrame([row], columns=service.raw_columns)
    S = service.preprocessing.transform(X)[service.features]
    probs = [service._base_proba(nm, S) for nm in service.base_order]
    return float(
        service.meta.predict_proba(np.asarray([probs], dtype=np.float64))[:, 1][0]
    )


def test_preprocessing_and_scoring_parity(service):
    events = _synthetic_events(service, N_ROWS)
    max_diff, n_alert = 0.0, 0
    for ev in events:
        wire = json.loads(json.dumps(ev, allow_nan=False, default=str))  # Kafka trip
        serving = service.score(wire)
        direct_p = _direct_proba(service, ev)
        assert serving["error"] is None
        diff = abs(serving["fraud_probability"] - direct_p)
        max_diff = max(max_diff, diff)
        n_alert += serving["is_fraud_pred"]
        assert diff <= TOL, (
            f"parity violated: serving={serving['fraud_probability']:.10f} "
            f"direct={direct_p:.10f} diff={diff:.2e}"
        )
    print(f"\nparity OK over {N_ROWS} rows | max |diff| = {max_diff:.2e} "
          f"| alerts = {n_alert}")


def test_threshold_agrees_with_artifact(service):
    with open(os.path.join(ARTIFACTS_DIR, "threshold.json")) as f:
        assert service.threshold == float(json.load(f)["threshold"])


def test_unseen_category_maps_to_reserved_code(service):
    cat = service.categorical_cols[0]
    row = {c: None for c in service.raw_columns}
    row[cat] = "definitely_never_seen_in_training_xyz"
    X = pd.DataFrame([row], columns=service.raw_columns)
    P = service.preprocessing.transform(X)
    assert float(P[cat].iloc[0]) == -1.0
