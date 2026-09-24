"""tools/make_reference.py — build artifacts/reference_distributions.json
from the TRAINING data, once, on a machine that has the train CSVs. Produces:
  - score reference: stacked-model scores on a sample of the holdout period
    (last 20% of the timeline), binned on [0,1] in 20 bins;
  - feature references: quantile-binned (numeric) or top-category (categorical)
    distributions of the drift features over the training window.

Usage:
  python tools/make_reference.py --data-dir <dir with train_*.csv> \
      [--artifacts-dir artifacts] [--sample 20000]

Without this file the exporter builds its reference from the first live events
(reference_bootstrap_used=1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "serving"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "monitoring", "exporter"))

import drift  # noqa: E402
from drift import DRIFT_FEATURES, CATEGORICAL_DRIFT, SCORE_BINS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--artifacts-dir", default=os.path.join(
        os.path.dirname(__file__), "..", "artifacts"))
    ap.add_argument("--sample", type=int, default=20_000)
    args = ap.parse_args()

    from scoring_udf import ScoringService  # heavy import after argparse
    svc = ScoringService(args.artifacts_dir)

    txn = pd.read_csv(os.path.join(args.data_dir, "train_transaction.csv"))
    idn = pd.read_csv(os.path.join(args.data_dir, "train_identity.csv"))
    df = txn.merge(idn, how="left", on="TransactionID")
    df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    del txn, idn

    n = len(df)
    train_part = df.iloc[: int(n * 0.8)]
    holdout = df.iloc[int(n * 0.8):]

    # ---- feature references over the training window -------------------------
    features: dict = {}
    for f in DRIFT_FEATURES:
        col = train_part[f] if f in train_part.columns else pd.Series(dtype=float)
        if f in CATEGORICAL_DRIFT:
            vals = [None if pd.isna(v) else str(v) for v in col]
            top = [c for c, _ in pd.Series(
                [v for v in vals if v is not None]
            ).value_counts().head(20).items()]
            features[f] = {"type": "categorical", "categories": top,
                           "props": drift.categorical_props(vals, top)}
        else:
            vals = [float(v) for v in col if not pd.isna(v)]
            edges = drift.quantile_edges(vals, 10)
            features[f] = {"type": "numeric", "edges": edges,
                           "props": drift.numeric_props(vals, edges)}
        print(f"[reference] {f}: {features[f]['type']}", flush=True)

    # ---- score reference: model scores on a holdout-period sample ------------
    rows = holdout.sample(n=min(args.sample, len(holdout)), random_state=42)
    scores = []
    for i, (_, row) in enumerate(rows.iterrows(), 1):
        ev = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        ev.pop("isFraud", None)
        r = svc.score_safe(ev)
        if r["fraud_probability"] is not None:
            scores.append(r["fraud_probability"])
        if i % 2000 == 0:
            print(f"[reference] scored {i}/{len(rows)}", flush=True)

    out = {"source": "training",
           "model_version": svc.model_version,
           "score": {"edges": SCORE_BINS,
                     "props": drift.numeric_props(scores, SCORE_BINS)},
           "features": features}
    path = os.path.join(args.artifacts_dir, "reference_distributions.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[reference] wrote {path} "
          f"({len(scores)} scores, {len(features)} features)", flush=True)


if __name__ == "__main__":
    main()
