import numpy as np
import pandas as pd

from credit_risk import config as C
from credit_risk.data import clean
from credit_risk.drift import psi
from credit_risk.features import RAW_FEATURES, engineer
from credit_risk.metrics import best_threshold, expected_cost


def _row(**kw):
    base = {"ID": 1, "LIMIT_BAL": 10000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
            "PAY_0": 0, **{f"PAY_{i}": 0 for i in range(2, 7)},
            **{f"BILL_AMT{i}": 1000 for i in range(1, 7)}, **{f"PAY_AMT{i}": 500 for i in range(1, 7)},
            "default payment next month": 0}
    base.update(kw)
    return base


def test_clean_merges_undocumented_codes():
    df = clean(pd.DataFrame([_row(EDUCATION=0), _row(EDUCATION=6, MARRIAGE=0)]))
    assert set(df.EDUCATION) == {4} and set(df.MARRIAGE) == {1, 3}
    assert "PAY_1" in df.columns and "ID" not in df.columns


def test_engineer_is_rowwise_and_finite():
    df = clean(pd.DataFrame([_row(), _row(PAY_0=3, BILL_AMT2=0)]))
    X = engineer(df)
    assert len(X) == 2 and np.isfinite(X.to_numpy(dtype=float)).all()
    assert X.loc[1, "max_delay"] == 3 and X.loc[1, "n_months_delayed"] == 1
    # transform of one row does not depend on the other rows
    pd.testing.assert_series_equal(engineer(df.iloc[[0]]).iloc[0], X.iloc[0], check_names=False)


def test_protected_attribute_is_not_a_feature():
    assert C.PROTECTED not in RAW_FEATURES
    assert C.PROTECTED not in engineer(clean(pd.DataFrame([_row()]))).columns


def test_cost_threshold_prefers_recall_when_fn_expensive():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    thr = best_threshold(y, p, cost_fn=5, cost_fp=1)
    assert thr < 0.5
    assert expected_cost(y, p, thr) <= expected_cost(y, p, 0.5)


def test_psi_zero_for_same_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, a) < 1e-6
    assert psi(a, a + 1.0) > 0.25
