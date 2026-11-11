"""Week 7 champion--challenger retraining replay for the churn capstone.

Run from the repository root, after the three monitoring stages::

    python run_weeks.py
    python -m monitor.performance
    python -m monitor.decision
    python -m monitor.retrain

This is a teaching simulation, not a production deployment workflow.  The
synthetic traffic is sampled from the original dataset; it is NOT an
independent live population.  Candidates are evaluated on labeled data that
were *not* used to train that candidate.  A passing challenger is staged as an
artifact for review, never automatically deployed to the API or GHCR.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train import FEATURES, TARGET, split

DATA = Path("data/churn.csv")
OUTPUT = Path("monitor/output")
MIN_POSITIVES = 60
WINDOW_WEEKS = 2
F1_IMPROVEMENT = 0.03
F1_FLOOR = 0.378  # baseline/metrics.json test F1 of 0.438 minus 0.060
MIN_RECALL = 0.65
MIN_PRECISION = 0.25


def make_model():
    """Use the same model family and preprocessing as Week 6."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def measure(model, dataframe):
    """Measure a model on one shared, unseen candidate-validation week."""
    y_true = dataframe[TARGET]
    pred = model.predict(dataframe[FEATURES])
    return {
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 3),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 3),
    }


def passes_gate(champion_metrics, challenger_metrics, positives):
    """Keep the existing champion unless the challenger passes all gates."""
    if positives < MIN_POSITIVES:
        return False, "Not enough confirmed churn cases to evaluate safely"
    if challenger_metrics["f1"] < F1_FLOOR:
        return False, "Challenger F1 below baseline protection floor"
    if challenger_metrics["recall"] < MIN_RECALL:
        return False, "Challenger recall below minimum"
    if challenger_metrics["precision"] < MIN_PRECISION:
        return False, "Challenger precision below minimum"
    if challenger_metrics["f1"] < champion_metrics["f1"] + F1_IMPROVEMENT:
        return False, "Challenger did not improve F1 enough over champion"
    return True, "Challenger passed all gates; staged for human review"


def load_week(week):
    path = OUTPUT / f"week_{week:02d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python run_weeks.py first.")
    data = pd.read_csv(path)
    if data.empty or not set(FEATURES + [TARGET]).issubset(data.columns):
        raise ValueError(f"Missing model features or labels in {path}")
    if data[FEATURES + [TARGET]].isna().any().any():
        raise ValueError(f"Missing values in {path}")
    if not data[TARGET].isin([0, 1]).all():
        raise ValueError(f"Invalid churn labels in {path}")
    return data


def main():
    decision_file = OUTPUT / "decisions.csv"
    if not decision_file.exists():
        raise FileNotFoundError(
            "Run python run_weeks.py, python -m monitor.performance, "
            "and python -m monitor.decision first."
        )

    # Reconstruct the Week 6 champion in this ephemeral GitHub Actions job.
    reference = pd.read_csv(DATA)
    (x_train, y_train), _, _ = split(reference)
    champion = make_model().fit(x_train, y_train)

    decisions = pd.read_csv(decision_file).sort_values("week")
    records = []
    for item in decisions.itertuples(index=False):
        current_week = int(item.week)
        if item.action != "retrain":
            continue

        # Labels arrive two weeks late; the newest labeled week is held out.
        # This avoids scoring the candidate on the records used to fit it.
        eval_week = current_week - 2
        train_weeks = list(range(eval_week - WINDOW_WEEKS, eval_week))
        if eval_week < 1 or min(train_weeks) < 1:
            raise ValueError("Not enough labeled training history yet")

        training_data = pd.concat(
            [load_week(week) for week in train_weeks], ignore_index=True
        )
        evaluation_data = load_week(eval_week)
        positives = int(evaluation_data[TARGET].sum())
        if training_data[TARGET].nunique() < 2:
            raise ValueError("Cannot fit challenger: only one target class")

        challenger = make_model().fit(training_data[FEATURES], training_data[TARGET])
        old_metrics = measure(champion, evaluation_data)
        new_metrics = measure(challenger, evaluation_data)
        accepted, reason = passes_gate(old_metrics, new_metrics, positives)

        if accepted:
            # Stage model for evaluation: do NOT change API/Docker or deploy.
            joblib.dump(challenger, OUTPUT / f"staged_challenger_week_{current_week:02d}.joblib")
            champion = challenger  # Only for subsequent weeks of this *simulation*.

        records.append({
            "monitoring_week": current_week,
            "training_weeks": "-".join(map(str, train_weeks)),
            "validation_week": eval_week,
            "training_rows": len(training_data),
            "validation_rows": len(evaluation_data),
            "positive_cases": positives,
            "champion_f1": old_metrics["f1"],
            "champion_precision": old_metrics["precision"],
            "champion_recall": old_metrics["recall"],
            "challenger_f1": new_metrics["f1"],
            "challenger_precision": new_metrics["precision"],
            "challenger_recall": new_metrics["recall"],
            "gate_passed": accepted,
            "decision": "stage_for_review" if accepted else "retain_champion",
            "reason": reason,
        })
        print(
            f"Week {current_week:02d}: trained on labeled weeks "
            f"{train_weeks}, validated on week {eval_week} | "
            f"champion F1={old_metrics['f1']:.3f}, "
            f"challenger F1={new_metrics['f1']:.3f} | "
            f"{'STAGED FOR REVIEW' if accepted else 'RETAIN CHAMPION'}: {reason}"
        )

    path = OUTPUT / "retraining_summary.csv"
    pd.DataFrame(records, columns=[
        "monitoring_week", "training_weeks", "validation_week",
        "training_rows", "validation_rows", "positive_cases",
        "champion_f1", "champion_precision", "champion_recall",
        "challenger_f1", "challenger_precision", "challenger_recall",
        "gate_passed", "decision", "reason",
    ]).to_csv(path, index=False)
    print(f"\nSaved champion-challenger report to {path}")
    print("No production deployment is performed by this script.")
    return records


if __name__ == "__main__":
    main()
