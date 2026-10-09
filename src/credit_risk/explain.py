"""Global + local explanations with SHAP (TreeExplainer on the tuned LightGBM).

    python -m credit_risk.explain
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import train_test_split

from . import config as C
from .data import load
from .features import engineer


def main(n_sample: int = 2000) -> dict:
    df = load()
    _, test_df = train_test_split(df, test_size=C.TEST_SIZE, stratify=df[C.TARGET], random_state=C.SEED)
    X = engineer(test_df).sample(n_sample, random_state=C.SEED)
    model = joblib.load(C.MODELS_DIR / "lgb_uncalibrated.joblib")

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv

    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(sv, X, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "shap_summary.png", dpi=150)
    plt.close()

    importance = np.abs(sv).mean(axis=0)
    order = np.argsort(importance)[::-1]
    top = [{"feature": X.columns[i], "mean_abs_shap": round(float(importance[i]), 4)} for i in order[:15]]
    (C.REPORTS_DIR / "shap_top_features.json").write_text(json.dumps(top, indent=2))
    for t in top:
        print(f"{t['feature']:22s} {t['mean_abs_shap']:.4f}")
    return {"top_features": top}


if __name__ == "__main__":
    main()
