"""Focused Week 7 tests for drift, retraining decisions, and safety gates."""

import numpy as np
import pandas as pd
import pytest

from monitor.decision import data_quality_ok, decide_action
from monitor.drift import FEATURES, drift_report, psi
from monitor.retrain import passes_gate


def test_psi_identical_input_is_zero():
    values = np.arange(1, 101)
    assert psi(values, values) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_numerical_shift():
    reference = np.linspace(10, 90, 200)
    current = reference + 25
    assert psi(reference, current) > 0.10


def test_psi_detects_categorical_shift():
    reference = [0] * 90 + [1] * 10
    current = [0] * 10 + [1] * 90
    assert psi(reference, current, categorical=True) > 0.25


def test_drift_report_rejects_missing_feature():
    reference = pd.DataFrame({name: [1, 2] for name in FEATURES})
    current = reference.drop(columns=[FEATURES[0]])
    with pytest.raises(ValueError, match="Missing feature"):
        drift_report(reference, current)


def test_bad_data_has_priority_over_retraining():
    action, _ = decide_action(
        quality_ok=False, f1=0.10, f1_floor=0.378, positives=100,
        drifted_features=3, max_psi=0.5,
    )
    assert action == "fix_data"


def test_f1_failure_with_enough_positives_requests_retraining():
    action, _ = decide_action(
        quality_ok=True, f1=0.20, f1_floor=0.378, positives=100,
        drifted_features=0, max_psi=0.01,
    )
    assert action == "retrain"


def test_low_f1_with_too_few_positives_does_not_retrain():
    action, _ = decide_action(
        quality_ok=True, f1=0.20, f1_floor=0.378, positives=20,
        drifted_features=0, max_psi=0.01,
    )
    assert action == "none"


def test_significant_drift_requires_investigation():
    action, _ = decide_action(
        quality_ok=True, f1=0.43, f1_floor=0.378, positives=100,
        drifted_features=2, max_psi=0.15,
    )
    assert action == "investigate"


def test_missing_labels_do_not_imply_retraining():
    action, _ = decide_action(
        quality_ok=True, f1=float("nan"), f1_floor=0.378,
        positives=float("nan"), drifted_features=0, max_psi=0.01,
    )
    assert action == "none"


def test_quality_check_rejects_invalid_churn_label():
    frame = pd.DataFrame({name: [1, 2] for name in FEATURES})
    frame["churned"] = [0, 2]
    assert not data_quality_ok(frame)


def test_quality_gate_rejects_challenger_below_floor():
    champion = {"f1": 0.22, "recall": 0.75, "precision": 0.28}
    challenger = {"f1": 0.30, "recall": 0.80, "precision": 0.35}
    accepted, reason = passes_gate(champion, challenger, positives=100)
    assert not accepted
    assert "floor" in reason.lower()


def test_quality_gate_accepts_strong_challenger_for_review():
    champion = {"f1": 0.40, "recall": 0.70, "precision": 0.29}
    challenger = {"f1": 0.48, "recall": 0.78, "precision": 0.36}
    accepted, reason = passes_gate(champion, challenger, positives=100)
    assert accepted
    assert "review" in reason.lower()


def test_quality_gate_needs_enough_positive_cases():
    champion = {"f1": 0.40, "recall": 0.70, "precision": 0.30}
    challenger = {"f1": 0.50, "recall": 0.80, "precision": 0.40}
    accepted, _ = passes_gate(champion, challenger, positives=10)
    assert not accepted
