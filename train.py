"""Train the churn model and write model/, model/metadata.json and metrics.json. Deterministic."""
import argparse
import hashlib
import json
import os
import platform

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["tenure_months", "monthly_charges", "contract_type", "support_calls_90d", "late_payments_12m",
            "has_security_addon", "age"]
TARGET = "churned"
SEED = 42


def split(df):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def scores(model, X, y):
    pred = model.predict(X)
    return {"precision": round(float(precision_score(y, pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
            "n": int(len(y)), "n_positive": int(y.sum())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/churn.csv")
    ap.add_argument("--out", default="model")
    ap.add_argument("--metrics", default="metrics.json")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.data)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split(df)
    model = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))]).fit(X_tr, y_tr)

    m2m = X_te["contract_type"] == 0
    metrics = {
        "validation": scores(model, X_va, y_va),
        "test": scores(model, X_te, y_te),
        "slices": {"month_to_month": scores(model, X_te[m2m], y_te[m2m]),
                   "long_contract": scores(model, X_te[~m2m], y_te[~m2m])},
        "rows": int(len(df)), "churn_rate": round(float(df[TARGET].mean()), 4),
        "data_md5": hashlib.md5(open(args.data, "rb").read()).hexdigest(),
    }
    meta = {"model_name": "churn-model", "version": "1.0.0", "features": FEATURES,
            "git_sha": os.getenv("GITHUB_SHA", "local")[:7], "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "test": {k: metrics["test"][k] for k in ("precision", "recall", "f1")}}
    os.makedirs(args.out, exist_ok=True)
    joblib.dump(model, os.path.join(args.out, "model.joblib"))
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    t = metrics["test"]
    print(f"trained on {len(X_tr)} rows | test precision {t['precision']} recall {t['recall']} f1 {t['f1']}")


if __name__ == "__main__":
    main()
