"""Unit tests for monitoring/exporter/drift.py (PSI, windows, bootstrap)."""
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "monitoring", "exporter"))

import drift
from drift import (DriftTracker, RollingWindow, psi, numeric_props,
                   categorical_props, quantile_edges, DRIFT_FEATURES)


def test_psi_zero_for_identical_distributions():
    p = [0.1, 0.2, 0.3, 0.4]
    assert abs(psi(p, p)) < 1e-12


def test_psi_positive_and_large_for_shifted_distributions():
    ref = [0.25, 0.25, 0.25, 0.25]
    shifted = [0.70, 0.10, 0.10, 0.10]
    v = psi(ref, shifted)
    assert v > 0.2  # above the PSI alert threshold


def test_numeric_props_clamps_out_of_range():
    props = numeric_props([-5.0, 0.5, 99.0], [0.0, 1.0, 2.0])
    assert abs(sum(props) - 1.0) < 1e-9
    assert props[0] > 0 and props[-1] > 0  # -5 -> first bin, 99 -> last bin


def test_categorical_props_other_bucket():
    props = categorical_props(["a", "b", "zzz", None], ["a", "b"])
    assert abs(sum(props) - 1.0) < 1e-9
    assert props[-1] > 0  # unseen 'zzz' lands in OTHER


def test_quantile_edges_monotone():
    edges = quantile_edges([float(i) for i in range(1000)], 10)
    assert all(b > a for a, b in zip(edges, edges[1:]))


def test_rolling_window_evicts_by_age():
    w = RollingWindow(max_age_s=10)
    w.add({"score": 0.1}, ts=100.0)
    w.add({"score": 0.9}, ts=109.0)
    assert w.snapshot("score", now=109.5) == [0.1, 0.9]
    assert w.snapshot("score", now=115.0) == [0.9]
    assert w.snapshot("score", now=125.0) == []


def test_bootstrap_freezes_and_scores_psi():
    t = DriftTracker(reference=None, bootstrap_n=200)
    import random
    random.seed(42)
    for _ in range(200):
        feats = {f: random.random() * 100 for f in DRIFT_FEATURES}
        feats["card6"] = random.choice(["debit", "credit"])
        t.observe_for_bootstrap(random.random() * 0.3, feats)
    assert t.ready() and t.reference["source"] == "bootstrap"
    # identical traffic -> tiny PSI
    same = [random.random() * 0.3 for _ in range(500)]
    assert t.score_psi(same) < 0.1
    # shifted traffic -> large PSI
    shifted = [0.7 + random.random() * 0.3 for _ in range(500)]
    assert t.score_psi(shifted) > 0.2
    # categorical drift
    assert t.feature_psi("card6", ["debit"] * 500) is not None


def test_psi_requires_minimum_sample():
    t = DriftTracker(reference={"score": {"edges": [0, 0.5, 1],
                                          "props": [0.5, 0.5]},
                                "features": {}})
    assert t.score_psi([0.1] * 50) is None      # <100 samples -> no signal
    assert t.score_psi([0.1] * 150) is not None
