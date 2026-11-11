"""Replay delayed-label performance checks for the Week 7 churn simulation.

Usage from the repository root (after `python run_weeks.py`):
    python monitor/performance.py

This is a teaching simulation. The weekly data is sampled from the original
customer data, so results are not an independent production holdout.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train import FEATURES, TARGET, split

REFERENCE = Path("data/churn.csv")
WEEKLY_DIR = Path("monitor/output")
BASELINE = Path("baseline/metrics.json")
LABEL_DELAY_WEEKS = 2
MIN_POSITIVES = 60
F1_TOLERANCE = 0.06


def evaluate_delayed_labels():
    reference = pd.read_csv(REFERENCE)
    (x_train, y_train), _, _ = split(reference)
    champion = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    champion.fit(x_train, y_train)
    with BASELINE.open(encoding="utf-8") as f:
        baseline_f1 = float(json.load(f)["test"]["f1"])
    f1_floor = max(0.0, baseline_f1 - F1_TOLERANCE)

    records = []
    for current_week in range(1, 15):
        labeled_week = current_week - LABEL_DELAY_WEEKS
        record = {"current_week": current_week, "labeled_week": "", "positives": "",
                  "precision": "", "recall": "", "f1": "", "f1_floor": round(f1_floor, 3),
                  "decision": "await_labels"}
        if labeled_week >= 1:
            weekly_path = WEEKLY_DIR / f"week_{labeled_week:02d}.csv"
            if not weekly_path.exists():
                raise FileNotFoundError(f"Missing {weekly_path}. Run python run_weeks.py first.")
            weekly = pd.read_csv(weekly_path)
            y_true = weekly[TARGET]
            y_pred = champion.predict(weekly[FEATURES])
            positives = int(y_true.sum())
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            record.update({"labeled_week": labeled_week, "positives": positives,
                           "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
                           "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
                           "f1": round(f1, 3),
                           "decision": ("insufficient_positives" if positives < MIN_POSITIVES else
                                        "review_retraining" if f1 < f1_floor else "healthy")})
        records.append(record)
        print(f"Week {current_week:02}: labeled week {record['labeled_week'] or '-'} | "
              f"F1={record['f1'] if record['f1'] != '' else '-'} | {record['decision']}")

    WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    output_path = WEEKLY_DIR / "performance_summary.csv"
    result.to_csv(output_path, index=False)
    print(f"\nSaved delayed-label performance report to {output_path}")
    return result


if __name__ == "__main__":
    evaluate_delayed_labels()
