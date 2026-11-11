"""Week 7 churn monitoring: decide when to fix, investigate, or retrain.

Run from the project root AFTER run_weeks.py and monitor.performance:
    python -m monitor.decision

This creates decisions.csv. It does not train or deploy a model.
"""

from pathlib import Path

import pandas as pd

from monitor.drift import FEATURES

OUTPUT_DIR = Path("monitor/output")
MIN_POSITIVES = 60
MODERATE_PSI = 0.10
SEVERE_PSI = 0.25  # Additional guardrail for one unusually large shift


def data_quality_ok(weekly):
    """Check essential feature/label fields before assessing model performance."""
    required = set(FEATURES) | {"churned"}
    if not required.issubset(weekly.columns):
        return False
    if weekly.empty or weekly[list(required)].isna().any().any():
        return False
    numeric = [name for name in FEATURES if name not in {"contract_type", "has_security_addon"}]
    for name in numeric:
        values = pd.to_numeric(weekly[name], errors="coerce")
        if values.isna().any() or (values < 0).any():
            return False
    if not weekly["churned"].isin([0, 1]).all():
        return False
    return True


def decide_action(*, quality_ok, f1, f1_floor, positives, drifted_features, max_psi):
    """Apply precedence: data quality, measured harm, drift, otherwise none."""
    if not quality_ok:
        return "fix_data", "Data validation failed; do not retrain on invalid data"
    if pd.notna(f1) and pd.notna(positives) and int(positives) >= MIN_POSITIVES:
        if float(f1) < float(f1_floor):
            return "retrain", "Labeled F1 below floor with sufficient positive cases"
    if int(drifted_features) >= 2 or float(max_psi) >= SEVERE_PSI:
        return "investigate", "Input drift detected; investigate before retraining"
    if pd.isna(f1):
        return "none", "Waiting for labels; continue monitoring"
    return "none", "No action threshold crossed"


def main():
    drift_path = OUTPUT_DIR / "summary.csv"
    performance_path = OUTPUT_DIR / "performance_summary.csv"
    if not drift_path.exists() or not performance_path.exists():
        raise FileNotFoundError(
            "First run: python run_weeks.py && python -m monitor.performance"
        )

    drift = pd.read_csv(drift_path)
    performance = pd.read_csv(performance_path)
    results = []
    for entry in drift.to_dict("records"):
        week = int(entry["week"])
        row = performance.loc[performance["current_week"] == week]
        if len(row) != 1:
            raise ValueError(f"Expected exactly one performance row for week {week}")
        perf = row.iloc[0]
        weekly_path = OUTPUT_DIR / f"week_{week:02d}.csv"
        if not weekly_path.exists():
            raise FileNotFoundError(weekly_path)
        quality_ok = data_quality_ok(pd.read_csv(weekly_path))
        action, reason = decide_action(
            quality_ok=quality_ok,
            f1=perf["f1"],
            f1_floor=perf["f1_floor"],
            positives=perf["positives"],
            drifted_features=entry["drifted_features"],
            max_psi=entry["max_psi"],
        )
        results.append({
            "week": week,
            "scenario": entry["scenario"],
            "max_psi": entry["max_psi"],
            "drifted_features": entry["drifted_features"],
            "labeled_week": perf["labeled_week"],
            "f1": perf["f1"],
            "action": action,
            "reason": reason,
        })
        print(f"Week {week:02d}: {action:11s} | {reason}")

    result = pd.DataFrame(results)
    output_file = OUTPUT_DIR / "decisions.csv"
    result.to_csv(output_file, index=False)
    print(f"\nSaved decision summary to {output_file}")
    return result


if __name__ == "__main__":
    main()
