"""Scoring: artifact loading + stacking-ensemble inference.

Two layers:
- ScoringService: plain Python, no Flink dependency. Loads the artifacts bundle
  once and scores one event dict at a time. Unit-testable via from_components().
- ScoreFunction: thin PyFlink MapFunction that owns a ScoringService
  (created in open(), i.e. models load once per task slot) and applies the
  fail-open rule: any inference error emits decision=REVIEW with an error flag —
  never a crash, never a silent drop.

The preprocessing pipeline is the same pickled object produced at training
time — preprocessing is not reimplemented here.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

import schemas

DECISION_ALERT = "ALERT"
DECISION_PASS = "PASS"
DECISION_REVIEW = "REVIEW"


class ScoringService:
    def __init__(self, artifacts_dir: str):
        import joblib  # lazy: keeps module importable in lightweight test envs

        a = artifacts_dir
        with open(os.path.join(a, "manifest.json")) as f:
            self.manifest = json.load(f)
        with open(os.path.join(a, "features.json")) as f:
            self.features: list[str] = json.load(f)["selected_features"]
        with open(os.path.join(a, "threshold.json")) as f:
            th = json.load(f)
        self.threshold: float = float(th["threshold"])
        self.base_order: list[str] = th["base_model_order_for_meta_features"]

        self.preprocessing = joblib.load(os.path.join(a, "preprocessing.pkl"))
        self.raw_columns: list[str] = list(self.preprocessing.feature_names_in_)
        self.categorical_cols: list[str] = list(
            self.manifest["data_contract"]["categorical_columns"]
        )
        self.numeric_cols: list[str] = [
            c for c in self.raw_columns if c not in set(self.categorical_cols)
        ]
        self.model_version: str = self.manifest["model_version"]
        self.bases = self._load_bases(a)
        self.meta = self._load_meta(a)

    # ---- native-format model loading -------------------------------------------
    @staticmethod
    def _load_bases(a: str) -> dict:
        import xgboost as xgb
        import lightgbm as lgb
        import catboost as cb

        x = xgb.XGBClassifier()
        x.load_model(os.path.join(a, "model_xgb.json"))
        l = lgb.Booster(model_file=os.path.join(a, "model_lgbm.txt"))
        c = cb.CatBoostClassifier()
        c.load_model(os.path.join(a, "model_catboost.cbm"))
        return {"xgb": x, "lgbm": l, "cat": c}

    @staticmethod
    def _load_meta(a: str):
        import xgboost as xgb

        m = xgb.XGBClassifier()
        m.load_model(os.path.join(a, "meta_learner.json"))
        return m

    @classmethod
    def from_components(
        cls,
        preprocessing,
        bases: dict,
        meta,
        features: list[str],
        threshold: float,
        categorical_cols: list[str],
        model_version: str = "test",
        base_order: Optional[list[str]] = None,
    ) -> "ScoringService":
        """Assemble a service from in-memory components (tests)."""
        self = cls.__new__(cls)
        self.preprocessing = preprocessing
        self.bases = bases
        self.meta = meta
        self.features = list(features)
        self.threshold = float(threshold)
        self.categorical_cols = list(categorical_cols)
        self.raw_columns = list(preprocessing.feature_names_in_)
        self.numeric_cols = [c for c in self.raw_columns
                             if c not in set(categorical_cols)]
        self.model_version = model_version
        self.base_order = base_order or ["xgb", "lgbm", "cat"]
        self.manifest = {"model_version": model_version}
        return self

    # ---- inference -----------------------------------------------------------
    def _base_proba(self, name: str, S: pd.DataFrame) -> float:
        m = self.bases[name]
        if hasattr(m, "predict_proba"):          # xgboost / catboost sklearn API
            return float(m.predict_proba(S)[:, 1][0])
        return float(np.asarray(m.predict(S)).ravel()[0])  # lightgbm Booster

    def score(self, event: dict) -> dict:
        """Score one normalized event dict. Raises on internal failure
        (fail-open policy is applied by the caller)."""
        t0 = time.time()
        tid = schemas.extract_transaction_id(event)
        row, warnings = schemas.coerce_and_complete(
            event, self.numeric_cols, self.categorical_cols
        )
        X = pd.DataFrame([row], columns=self.raw_columns)
        P = self.preprocessing.transform(X)
        S = P[self.features]
        base_probs = [self._base_proba(nm, S) for nm in self.base_order]
        meta_X = np.asarray([base_probs], dtype=np.float64)
        proba = float(self.meta.predict_proba(meta_X)[:, 1][0])
        is_fraud = int(proba >= self.threshold)
        return {
            "transaction_id": tid,
            "fraud_probability": proba,
            "is_fraud_pred": is_fraud,
            "decision": DECISION_ALERT if is_fraud else DECISION_PASS,
            "model_version": self.model_version,
            "processing_ts": time.time(),
            "latency_ms": round((time.time() - t0) * 1000.0, 3),
            "error": None,
            "warnings": warnings or None,
        }

    def score_safe(self, event: dict) -> dict:
        """Fail-open wrapper: errors -> REVIEW, never a drop."""
        try:
            return self.score(event)
        except Exception as e:
            return {
                "transaction_id": schemas.extract_transaction_id(event),
                "fraud_probability": None,
                "is_fraud_pred": None,
                "decision": DECISION_REVIEW,
                "model_version": getattr(self, "model_version", None),
                "processing_ts": time.time(),
                "latency_ms": None,
                "error": f"{type(e).__name__}: {e}"[:500],
                "warnings": None,
            }


def _score_record(service: ScoringService, raw: str) -> str:
    """Kafka string in -> scored JSON string out (validation ran upstream;
    parsing here is defensive only)."""
    event, err = schemas.parse_event(raw)
    if event is None:
        result = {
            "transaction_id": None,
            "fraud_probability": None,
            "is_fraud_pred": None,
            "decision": DECISION_REVIEW,
            "model_version": service.model_version,
            "processing_ts": time.time(),
            "latency_ms": None,
            "error": err,
            "warnings": None,
        }
    else:
        result = service.score_safe(event)
    return json.dumps(result, allow_nan=False)


try:  # Flink is absent in unit-test environments; the service above still works.
    # PyFlink 2.x removed RichMapFunction: MapFunction itself has open()/close().
    try:
        from pyflink.datastream import RichMapFunction as _BaseMapFunction  # 1.x
    except ImportError:
        from pyflink.datastream import MapFunction as _BaseMapFunction  # 2.x

    class ScoreFunction(_BaseMapFunction):
        """Single scoring operator; models loaded once in open()."""

        def __init__(self, artifacts_dir: str):
            self.artifacts_dir = artifacts_dir
            self.service: Optional[ScoringService] = None

        def open(self, runtime_context):
            self.service = ScoringService(self.artifacts_dir)

        def map(self, raw: str) -> str:
            return _score_record(self.service, raw)

except ImportError:  # pragma: no cover
    ScoreFunction = None  # type: ignore
