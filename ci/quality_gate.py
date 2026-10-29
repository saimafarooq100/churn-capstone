"""Model quality gate: exit code 1 blocks the pipeline.

    python ci/quality_gate.py        # reads metrics.json and baseline/metrics.json
"""
import json
import sys

# The clean model has 0.754 recall and 0.438 F1, while shuffled labels give much worse F1.
# A recall floor of 0.65 protects churners; precision and slice floors catch weaker models.
# F1 may move a little with the test sample, so I allow a 0.06 reduction, not a large drop.
MIN_TEST_RECALL = 0.65
MIN_TEST_PRECISION = 0.25
F1_TOLERANCE = 0.06
MIN_SLICE_RECALL = {"month_to_month": 0.70, "long_contract": 0.15}
MIN_SLICE_POSITIVES = 20        # fewer churners than this in a slice: too noisy to enforce


def evaluate(candidate, baseline):
    """Return a list of (rule, passed, observed, required)."""
    t = candidate["test"]
    need_f1 = round(baseline["test"]["f1"] - F1_TOLERANCE, 3)
    rules = [
        ("test recall >= floor", t["recall"] >= MIN_TEST_RECALL, t["recall"], f">= {MIN_TEST_RECALL}"),
        ("test precision >= floor", t["precision"] >= MIN_TEST_PRECISION, t["precision"], f">= {MIN_TEST_PRECISION}"),
        ("test F1 not below baseline - tolerance", t["f1"] >= need_f1, t["f1"], f">= {need_f1}"),
    ]
    for name, floor in MIN_SLICE_RECALL.items():
        s = candidate["slices"][name]
        if s["n_positive"] < MIN_SLICE_POSITIVES:
            rules.append((f"{name} slice recall", True, s["recall"], f"skipped, only {s['n_positive']} churners"))
        else:
            rules.append((f"{name} slice recall >= floor", s["recall"] >= floor, s["recall"], f">= {floor}"))
    return rules


def main():
    with open("metrics.json") as f:
        candidate = json.load(f)
    with open("baseline/metrics.json") as f:
        baseline = json.load(f)
    rules = evaluate(candidate, baseline)
    width = max(len(r[0]) for r in rules)
    for rule, ok, observed, required in rules:
        print(f"{'PASS' if ok else 'FAIL'}  {rule.ljust(width)}  {observed:<8}  {required}")
    if all(ok for _, ok, _, _ in rules):
        print("quality gate PASSED")
        return 0
    print("quality gate FAILED: this model must not be shipped")
    return 1


if __name__ == "__main__":
    sys.exit(main())
