import numpy as np
import pandas as pd

from ci import validate_data


def good_frame(n=1500):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n), "monthly_charges": rng.uniform(20, 120, n) + rng.random(n) * 1e-3,
        "contract_type": rng.integers(0, 3, n), "support_calls_90d": rng.poisson(1, n), "late_payments_12m": rng.poisson(0.5, n),
        "has_security_addon": rng.integers(0, 2, n), "age": rng.integers(18, 80, n),
        "churned": (rng.random(n) < 0.17).astype(int),
    })


def failed(df):
    return {name for name, ok, _ in validate_data.run_checks(df) if not ok}


def test_a_clean_frame_passes():
    assert failed(good_frame()) == set()


def test_negative_tenure_is_caught():
    assert "tenure is within 0 to 120 months" in failed(good_frame().assign(tenure_months=-3))


def test_unknown_contract_is_caught():
    assert "contract type is 0, 1 or 2" in failed(good_frame().assign(contract_type=5))


def test_wrong_label_is_caught():
    assert "churn label is binary" in failed(good_frame().assign(churned=2))


def test_invalid_age_is_caught():
    assert "age is within 18 to 110" in failed(good_frame().assign(age=7))


def test_missing_monthly_bill_is_caught():
    frame = good_frame()
    frame.loc[0, "monthly_charges"] = np.nan
    assert "no missing values" in failed(frame)


def test_unrealistic_churn_rate_is_caught():
    assert "churn rate is plausible" in failed(good_frame().assign(churned=1))


def test_duplicate_rows_are_caught():
    frame = good_frame()
    frame.iloc[100:300] = frame.iloc[0:200].to_numpy()
    assert "duplicate share is acceptable" in failed(frame)
