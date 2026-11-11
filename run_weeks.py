"""Generate 14 weeks of *simulated* churn traffic and a PSI drift report.

This is a teaching simulation, not actual customer activity. Weeks 5-8 mimic a
change in incoming features; weeks 9-14 mimic concept drift by scrambling
churn labels while leaving the feature distribution largely unchanged.

Run from the churn-capstone repository root:
    python run_weeks.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from monitor.drift import FEATURES, drift_report

REFERENCE = Path("data/churn.csv")
OUTPUT_DIR = Path("monitor/output")
ROWS_PER_WEEK = 2000
DRIFT_PSI_THRESHOLD = 0.10  # Illustrative investigation threshold


def generate_weeks(reference, output_dir=OUTPUT_DIR):
    """Write synthetic weekly CSVs and a compact drift summary."""
    if reference.empty or not {"churned", *FEATURES}.issubset(reference.columns):
        raise ValueError("Expected churn.csv with the seven model features and churned")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for week in range(1, 15):
        rng = np.random.default_rng(2026 + week)
        indices = rng.choice(len(reference), size=ROWS_PER_WEEK, replace=True)
        current = reference.iloc[indices].copy().reset_index(drop=True)

        if 5 <= week <= 8:
            # Data drift: charges become higher than in the training data.
            current["monthly_charges"] = (
                current["monthly_charges"].add(28.0).clip(upper=150.0).round(2)
            )
            scenario = "input_drift"
        elif week >= 9:
            # Concept-drift teaching example: the link between inputs and
            # churn outcomes changes, while the input data stays similar.
            current["churned"] = rng.permutation(current["churned"].to_numpy())
            scenario = "concept_drift"
        else:
            scenario = "stable"

        current.to_csv(output_dir / f"week_{week:02d}.csv", index=False)
        psi_values = drift_report(reference, current)
        drifted_features = [
            name for name, value in psi_values.items()
            if value >= DRIFT_PSI_THRESHOLD
        ]
        result = {
            "week": week,
            "scenario": scenario,
            "rows": len(current),
            "churn_rate": round(float(current["churned"].mean()), 4),
            "max_psi": round(max(psi_values.values()), 4),
            "drifted_features": len(drifted_features),
            "monthly_charges_psi": psi_values["monthly_charges"],
        }
        summary.append(result)
        print(
            f"Week {week:02d}: {scenario:13s} | "
            f"monthly_charges PSI={result['monthly_charges_psi']:.3f} | "
            f"drifted features={len(drifted_features)}"
        )

    pd.DataFrame(summary).to_csv(output_dir / "summary.csv", index=False)
    print(f"\nSaved 14 weekly datasets and summary.csv in {output_dir}/")
    print("Note: PSI checks input drift only; it cannot directly reveal concept drift.")
    return summary


if __name__ == "__main__":
    if not REFERENCE.exists():
        raise SystemExit("Run this script from your churn-capstone repository root.")
    data = pd.read_csv(REFERENCE)
    generate_weeks(data)
