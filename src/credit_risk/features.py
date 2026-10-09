"""Row-wise feature engineering.

Every feature uses only the customer's own six-month history, so the transform
is stateless and cannot leak information across the train/test split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

RAW_FEATURES = ["LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE", *C.PAY_STATUS, *C.BILL_AMT, *C.PAY_AMT]


def _safe_div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), 0.0)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df[RAW_FEATURES].copy()
    pay = df[C.PAY_STATUS].to_numpy()
    bill = df[C.BILL_AMT].to_numpy(dtype=float)
    paid = df[C.PAY_AMT].to_numpy(dtype=float)
    limit = df["LIMIT_BAL"].to_numpy(dtype=float)

    # Delinquency behaviour
    out["n_months_delayed"] = (pay > 0).sum(axis=1)
    out["max_delay"] = pay.max(axis=1)
    out["recent_delay"] = pay[:, 0]
    out["delay_trend"] = pay[:, :3].mean(axis=1) - pay[:, 3:].mean(axis=1)  # >0 = getting worse

    # Credit utilisation
    util = bill / limit[:, None]
    out["util_latest"] = util[:, 0]
    out["util_mean"] = util.mean(axis=1)
    out["util_max"] = util.max(axis=1)

    # Repayment behaviour: amount paid in month i settles bill of month i+1
    ratios = _safe_div(paid[:, :5], bill[:, 1:])
    out["pay_ratio_mean"] = np.clip(ratios, 0, 5).mean(axis=1)
    out["pay_ratio_latest"] = np.clip(ratios[:, 0], 0, 5)
    out["months_zero_payment"] = (paid == 0).sum(axis=1)
    out["bill_growth"] = _safe_div(bill[:, 0] - bill[:, 5], np.abs(bill[:, 5]) + 1)
    out["avg_payment"] = paid.mean(axis=1)
    return out


def split_xy(df: pd.DataFrame, engineered: bool = True):
    X = engineer(df) if engineered else df[RAW_FEATURES].copy()
    return X, df[C.TARGET].to_numpy()
