"""Population Stability Index (PSI) drift monitoring.

Compares a reference window (training data) to a new batch, feature by feature.
Rule of thumb used in credit risk: PSI < 0.1 stable, 0.1-0.25 watch, > 0.25 act.

    python -m credit_risk.drift

The report checks two batches:
  * the real held-out test set (should be stable: same population), and
  * a *simulated* stress scenario (documented as such) where repayment delays and
    utilisation rise, to show that the monitor actually fires.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C
from .data import load
from .features import engineer

MONITORED = ["LIMIT_BAL", "AGE", "recent_delay", "n_months_delayed", "util_latest", "util_mean",
             "pay_ratio_mean", "months_zero_payment", "avg_payment"]


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """PSI with quantile bins from the reference; handles low-cardinality features."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant or discrete feature -> use its distinct values
        cats = np.union1d(np.unique(ref), np.unique(cur))
        r = np.array([(ref == c).mean() for c in cats])
        c_ = np.array([(cur == c).mean() for c in cats])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c_ = np.histogram(cur, edges)[0] / len(cur)
    r, c_ = np.clip(r, 1e-4, None), np.clip(c_, 1e-4, None)
    return float(np.sum((c_ - r) * np.log(c_ / r)))


def status(v: float) -> str:
    return "stable" if v < 0.1 else ("watch" if v < 0.25 else "ALERT")


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, cols=MONITORED) -> dict:
    return {c: {"psi": round(psi(ref[c].to_numpy(), cur[c].to_numpy()), 4),
                "status": status(psi(ref[c].to_numpy(), cur[c].to_numpy()))} for c in cols}


def simulate_stress(df: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """SIMULATED scenario: 30% of customers slip one more month on every payment and
    carry 25% higher balances. Used only to demonstrate alerting."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out[C.BILL_AMT] = out[C.BILL_AMT].astype(float)
    hit = rng.random(len(out)) < 0.30
    for c in C.PAY_STATUS:
        out.loc[hit, c] = np.clip(out.loc[hit, c] + 1, -2, 8)
    for c in C.BILL_AMT:
        out.loc[hit, c] = out.loc[hit, c] * 1.25
    return out


def main() -> dict:
    df = load()
    train_df, test_df = train_test_split(df, test_size=C.TEST_SIZE, stratify=df[C.TARGET], random_state=C.SEED)
    ref = engineer(train_df)
    report = {
        "test_set_vs_train": drift_report(ref, engineer(test_df)),
        "simulated_stress_vs_train": drift_report(ref, engineer(simulate_stress(test_df))),
    }
    (C.REPORTS_DIR / "drift_report.json").write_text(json.dumps(report, indent=2))
    for k, rep in report.items():
        print(f"\n{k}")
        for f, v in rep.items():
            print(f"  {f:22s} PSI={v['psi']:.4f}  {v['status']}")
    return report


if __name__ == "__main__":
    main()
