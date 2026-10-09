"""Evaluation helpers: ranking metrics, cost-based threshold, bootstrap CIs, fairness."""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config as C


def ks_stat(y, p) -> float:
    """Kolmogorov-Smirnov separation between score distributions (standard in credit risk)."""
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def ranking_metrics(y, p) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": ks_stat(y, p),
        "brier": float(brier_score_loss(y, p)),
    }


def expected_cost(y, p, thr, cost_fn=C.COST_FN, cost_fp=C.COST_FP) -> float:
    pred = p >= thr
    fn = np.sum((y == 1) & ~pred)
    fp = np.sum((y == 0) & pred)
    return float((cost_fn * fn + cost_fp * fp) / len(y))


def best_threshold(y, p, cost_fn=C.COST_FN, cost_fp=C.COST_FP) -> float:
    grid = np.linspace(0.02, 0.98, 97)
    costs = [expected_cost(y, p, t, cost_fn, cost_fp) for t in grid]
    return float(grid[int(np.argmin(costs))])


def threshold_metrics(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "flag_rate": float(pred.mean()),
        "expected_cost_per_customer": expected_cost(y, p, thr),
    }


def bootstrap_ci(y, p, fn=roc_auc_score, n=1000, seed=C.SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(fn(y[b], p[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def group_report(y, p, thr, groups) -> dict:
    """Per-group flag rate, recall (TPR) and false-positive rate."""
    out = {}
    for g in np.unique(groups):
        m = groups == g
        pred = p[m] >= thr
        yg = y[m]
        out[str(g)] = {
            "n": int(m.sum()),
            "base_default_rate": float(yg.mean()),
            "flag_rate": float(pred.mean()),
            "tpr": float(pred[yg == 1].mean()),
            "fpr": float(pred[yg == 0].mean()),
            "roc_auc": float(roc_auc_score(yg, p[m])),
        }
    return out
