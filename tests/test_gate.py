from ci import quality_gate

BASELINE = {"test": {"precision": 0.308, "recall": 0.754, "f1": 0.438}}
GOOD = {"test": {"precision": 0.308, "recall": 0.754, "f1": 0.438},
        "slices": {"month_to_month": {"recall": 0.899, "n_positive": 129}, "long_contract": {"recall": 0.263, "n_positive": 38}}}


def failed(candidate):
    return [r for r, ok, _, _ in quality_gate.evaluate(candidate, BASELINE) if not ok]


def test_the_baseline_itself_passes():
    assert failed(GOOD) == []


def test_large_recall_drop_is_blocked():
    candidate = {**GOOD, "test": {**GOOD["test"], "recall": 0.50}}
    assert "test recall >= floor" in failed(candidate)


def test_a_small_f1_change_is_allowed():
    candidate = {**GOOD, "test": {**GOOD["test"], "f1": GOOD["test"]["f1"] - 0.01}}
    assert failed(candidate) == []


def test_large_f1_drop_is_blocked():
    candidate = {**GOOD, "test": {**GOOD["test"], "f1": GOOD["test"]["f1"] - 0.12}}
    assert "test F1 not below baseline - tolerance" in failed(candidate)


def test_a_small_slice_is_skipped():
    candidate = {**GOOD, "slices": {**GOOD["slices"], "long_contract": {"recall": 0, "n_positive": 4}}}
    assert failed(candidate) == []


def test_a_large_slice_regression_is_blocked():
    candidate = {**GOOD, "slices": {**GOOD["slices"], "month_to_month": {"recall": 0.55, "n_positive": 129}}}
    assert "month_to_month slice recall >= floor" in failed(candidate)
