
"""Population Stability Index (PSI) for weekly customer-churn monitoring."""

import numpy as np
import pandas as pd

FEATURES = [
    "tenure_months", "monthly_charges", "contract_type",
    "support_calls_90d", "late_payments_12m",
    "has_security_addon", "age",
]

CATEGORICAL = {"contract_type", "has_security_addon"}


def psi(reference, current, categorical=False, bins=10):
    """Compare current data with the training reference."""
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()

    if ref.empty or cur.empty:
        raise ValueError("PSI needs non-empty data")

    if categorical:
        ref_share = ref.value_counts(normalize=True)
        cur_share = cur.value_counts(normalize=True)
        groups = ref_share.index.union(cur_share.index)
        expected = ref_share.reindex(groups, fill_value=0).to_numpy()
        actual = cur_share.reindex(groups, fill_value=0).to_numpy()
    else:
        baseline = ref.to_numpy(dtype=float)
        observed = cur.to_numpy(dtype=float)
        edges = np.unique(
            np.quantile(baseline, np.linspace(0, 1, bins + 1))
        )

        if len(edges) == 1:
            value = edges[0]
            edges = np.array([
                -np.inf, value - 0.5, value + 0.5, np.inf
            ])
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf

        expected = np.histogram(baseline, bins=edges)[0] / len(baseline)
        actual = np.histogram(observed, bins=edges)[0] / len(observed)

    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)

    return float(
        np.sum((actual - expected) * np.log(actual / expected))
    )


def drift_report(reference_df, current_df):
    """Calculate PSI for the seven churn model features."""
    for feature in FEATURES:
        if feature not in reference_df or feature not in current_df:
            raise ValueError(f"Missing feature: {feature}")

    return {
        feature: round(
            psi(
                reference_df[feature],
                current_df[feature],
                categorical=feature in CATEGORICAL,
            ),
            4,
        )
        for feature in FEATURES
    }

