"""Unit tests for ScoringService using injected fake components — verifies
threshold application, output schema, and the fail-open rule without needing
the real artifacts bundle or the booster libraries."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "serving"))

import numpy as np
import pandas as pd

import scoring_udf
from scoring_udf import ScoringService, DECISION_ALERT, DECISION_PASS, DECISION_REVIEW

RAW_COLS = ["TransactionAmt", "C1", "card4"]
FEATURES = ["TransactionAmt", "C1"]


class FakePreprocessing:
    feature_names_in_ = np.array(RAW_COLS)
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["TransactionAmt"] = pd.to_numeric(out["TransactionAmt"]).fillna(0.0)
        out["C1"] = pd.to_numeric(out["C1"]).fillna(0.0)
        out["card4"] = 0.0
        return out


class FakeBase:
    def __init__(self, p): self.p = p
    def predict_proba(self, S): return np.array([[1 - self.p, self.p]])


class FakeLgbmBooster:  # lightgbm Booster has predict(), not predict_proba()
    def __init__(self, p): self.p = p
    def predict(self, S): return np.array([self.p])


class FakeMeta:
    """Meta returns the mean of the three base probabilities."""
    def predict_proba(self, meta_X):
        p = float(np.mean(meta_X))
        return np.array([[1 - p, p]])


class ExplodingBase:
    def predict_proba(self, S): raise RuntimeError("boom")


def make_service(pxgb=0.9, plgbm=0.9, pcat=0.9, threshold=0.5):
    return ScoringService.from_components(
        preprocessing=FakePreprocessing(),
        bases={"xgb": FakeBase(pxgb), "lgbm": FakeLgbmBooster(plgbm),
               "cat": FakeBase(pcat)},
        meta=FakeMeta(), features=FEATURES, threshold=threshold,
        categorical_cols=["card4"], model_version="test-1")


def test_alert_above_threshold():
    r = make_service(0.9, 0.9, 0.9, threshold=0.5).score(
        {"TransactionID": 1, "TransactionAmt": 10.0})
    assert r["is_fraud_pred"] == 1 and r["decision"] == DECISION_ALERT
    assert abs(r["fraud_probability"] - 0.9) < 1e-9
    assert r["transaction_id"] == "1" and r["model_version"] == "test-1"
    assert r["error"] is None and r["latency_ms"] is not None


def test_pass_below_threshold():
    r = make_service(0.1, 0.1, 0.1, threshold=0.5).score({"TransactionID": 2})
    assert r["is_fraud_pred"] == 0 and r["decision"] == DECISION_PASS


def test_threshold_boundary_inclusive():
    r = make_service(0.5, 0.5, 0.5, threshold=0.5).score({"TransactionID": 3})
    assert r["is_fraud_pred"] == 1  # proba >= threshold


def test_fail_open_on_inference_error():
    svc = make_service()
    svc.bases["xgb"] = ExplodingBase()
    r = svc.score_safe({"TransactionID": 4, "TransactionAmt": 1.0})
    assert r["decision"] == DECISION_REVIEW
    assert r["fraud_probability"] is None and r["is_fraud_pred"] is None
    assert "RuntimeError" in r["error"]
    assert r["transaction_id"] == "4"          # event id survives the failure


def test_score_record_end_to_end_json():
    svc = make_service()
    out = json.loads(scoring_udf._score_record(
        svc, json.dumps({"TransactionID": 9, "TransactionAmt": 3.5})))
    assert out["decision"] in (DECISION_ALERT, DECISION_PASS)
    out2 = json.loads(scoring_udf._score_record(svc, "{{{not json"))
    assert out2["decision"] == DECISION_REVIEW and out2["error"]


def test_base_order_respected():
    svc = make_service(0.9, 0.5, 0.1)
    seen = []
    class Spy:
        def __init__(self, name, p): self.name, self.p = name, p
        def predict_proba(self, S):
            seen.append(self.name); return np.array([[1 - self.p, self.p]])
    svc.bases = {"xgb": Spy("xgb", 0.9), "lgbm": Spy("lgbm", 0.5),
                 "cat": Spy("cat", 0.1)}
    svc.score({"TransactionID": 1})
    assert seen == svc.base_order == ["xgb", "lgbm", "cat"]
