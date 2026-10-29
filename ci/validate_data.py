"""Data checks that run BEFORE training. Exit code 1 if any check fails.

    python ci/validate_data.py data/churn.csv
"""
import sys

import pandas as pd

COLUMNS = ["tenure_months", "monthly_charges", "contract_type", "support_calls_90d", "late_payments_12m",
           "has_security_addon", "age", "churned"]
MIN_ROWS = 1000
CHURN_RATE_RANGE = (0.05, 0.40)        # Allows normal variation around the expected 17% churn rate.
MAX_DUPLICATE_SHARE = 0.05


def run_checks(df):
    """Return a list of (name, passed, detail)."""
    results = []

    def check(name, passed, detail=""):
        results.append((name, bool(passed), detail))

    check("columns match the contract", list(df.columns) == COLUMNS, f"found {list(df.columns)}")
    if list(df.columns) != COLUMNS:
        return results
    check("enough rows", len(df) >= MIN_ROWS, f"{len(df)} rows (minimum {MIN_ROWS})")
    check("no missing values", df.isna().sum().sum() == 0, f"{int(df.isna().sum().sum())} missing")
    # I check the valid ranges before training so bad customer records cannot affect the model.
    check("tenure is within 0 to 120 months", df["tenure_months"].between(0, 120).all(), "tenure must be 0 to 120")
    check("monthly charges are positive", (df["monthly_charges"] > 0).all(), "monthly bill must be above zero")
    check("contract type is 0, 1 or 2", df["contract_type"].isin([0, 1, 2]).all(), "only three contract types allowed")
    check("support call counts are non-negative", (df["support_calls_90d"] >= 0).all(), "calls cannot be negative")
    check("late payment counts are non-negative", (df["late_payments_12m"] >= 0).all(), "payments cannot be negative")
    check("security add-on is binary", df["has_security_addon"].isin([0, 1]).all(), "only zero or one")
    check("age is within 18 to 110", df["age"].between(18, 110).all(), "age must be 18 to 110")
    check("churn label is binary", df["churned"].isin([0, 1]).all(), "target must be zero or one")
    churn_rate = df["churned"].mean()
    check("churn rate is plausible", CHURN_RATE_RANGE[0] <= churn_rate <= CHURN_RATE_RANGE[1],
          f"rate {churn_rate:.3f} must be between {CHURN_RATE_RANGE[0]} and {CHURN_RATE_RANGE[1]}")
    duplicate_share = df.duplicated().mean()
    check("duplicate share is acceptable", duplicate_share <= MAX_DUPLICATE_SHARE,
          f"{duplicate_share:.1%} duplicates (maximum {MAX_DUPLICATE_SHARE:.0%})")

    return results


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    df = pd.read_csv(argv[0] if argv else "data/churn.csv")
    results = run_checks(df)
    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print(f"\n{len(failed)} data check(s) failed: the model was NOT trained.")
        return 1
    print(f"\nall {len(results)} data checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
