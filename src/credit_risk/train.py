"""Run every experiment, pick a winner on cross-validation, evaluate once on the held-out test set.

    python -m credit_risk.train            # full run (includes Optuna tuning)
    python -m credit_risk.train --trials 5 # quicker

Protocol
--------
1. Stratified 80/20 split. The 20% test set is touched exactly once, at the end.
2. Every candidate is scored with 5-fold stratified CV on the training set.
3. The decision threshold is chosen on out-of-fold training predictions (never on test).
4. All runs are logged to MLflow (sqlite:///mlflow.db); a summary is written to reports/results.json.
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config as C
from . import metrics as M
from .data import load
from .features import RAW_FEATURES, engineer

optuna.logging.set_verbosity(optuna.logging.WARNING)


def candidates(tuned_lgb_params: dict | None = None) -> dict:
    """name -> (estimator, uses_engineered_features)"""
    c = {
        "dummy_prior": (DummyClassifier(strategy="prior"), False),
        "logreg_raw": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), False),
        "logreg_engineered": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), True),
        "lightgbm_default": (lgb.LGBMClassifier(random_state=C.SEED, verbose=-1), True),
        "xgboost_default": (
            XGBClassifier(random_state=C.SEED, eval_metric="logloss", n_jobs=2, tree_method="hist"),
            True,
        ),
    }
    if tuned_lgb_params:
        c["lightgbm_tuned"] = (lgb.LGBMClassifier(**tuned_lgb_params), True)
    return c


def tune_lightgbm(X, y, n_trials: int) -> dict:
    skf = StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=C.SEED)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "random_state": C.SEED,
            "verbose": -1,
        }
        p = cross_val_predict(lgb.LGBMClassifier(**params), X, y, cv=skf, method="predict_proba")[:, 1]
        return M.ranking_metrics(y, p)["roc_auc"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params, subsample_freq=1, random_state=C.SEED, verbose=-1)
    return best


def main(n_trials: int = 30) -> dict:
    df = load()
    train_df, test_df = train_test_split(df, test_size=C.TEST_SIZE, stratify=df[C.TARGET], random_state=C.SEED)
    y_tr, y_te = train_df[C.TARGET].to_numpy(), test_df[C.TARGET].to_numpy()
    X_tr_raw, X_te_raw = train_df[RAW_FEATURES], test_df[RAW_FEATURES]
    X_tr_eng, X_te_eng = engineer(train_df), engineer(test_df)

    mlflow.set_tracking_uri(C.MLRUNS_URI)
    mlflow.set_experiment("credit-default-risk")
    skf = StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=C.SEED)

    t0 = time.time()
    tuned = tune_lightgbm(X_tr_eng, y_tr, n_trials) if n_trials > 0 else None
    tune_seconds = time.time() - t0

    cv_results, oof = {}, {}
    for name, (est, eng) in candidates(tuned).items():
        X = X_tr_eng if eng else X_tr_raw
        with mlflow.start_run(run_name=name):
            p = cross_val_predict(est, X, y_tr, cv=skf, method="predict_proba")[:, 1]
            m = M.ranking_metrics(y_tr, p)
            mlflow.log_params({"model": name, "engineered_features": eng, "n_features": X.shape[1]})
            if name == "lightgbm_tuned":
                mlflow.log_params({f"lgb_{k}": v for k, v in tuned.items()})
            mlflow.log_metrics({f"cv_{k}": v for k, v in m.items()})
        cv_results[name] = m
        oof[name] = p
        print(f"{name:20s} CV ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  KS={m['ks']:.4f}")

    best_name = max((n for n in cv_results if n != "dummy_prior"), key=lambda n: cv_results[n]["roc_auc"])
    best_est, best_eng = candidates(tuned)[best_name]
    print(f"\nSelected on CV: {best_name}")

    # Calibrate the winner (isotonic, internal 5-fold) so scores read as probabilities.
    final = CalibratedClassifierCV(best_est, method="isotonic", cv=C.N_FOLDS)
    X_fit, X_eval = (X_tr_eng, X_te_eng) if best_eng else (X_tr_raw, X_te_raw)
    oof_cal = cross_val_predict(
        CalibratedClassifierCV(best_est, method="isotonic", cv=C.N_FOLDS), X_fit, y_tr, cv=skf, method="predict_proba"
    )[:, 1]
    thr = M.best_threshold(y_tr, oof_cal)
    final.fit(X_fit, y_tr)

    # ---- the one and only look at the test set ----
    p_te = final.predict_proba(X_eval)[:, 1]
    base = candidates(tuned)["logreg_raw"][0].fit(X_tr_raw, y_tr)
    p_base = base.predict_proba(X_te_raw)[:, 1]

    test = {
        "selected_model": best_name,
        "final": {
            **M.ranking_metrics(y_te, p_te),
            "roc_auc_95ci": M.bootstrap_ci(y_te, p_te),
            "at_cost_threshold": M.threshold_metrics(y_te, p_te, thr),
            "at_0.5": M.threshold_metrics(y_te, p_te, 0.5),
        },
        "baseline_logreg_raw": {
            **M.ranking_metrics(y_te, p_base),
            "roc_auc_95ci": M.bootstrap_ci(y_te, p_base),
            "at_0.5": M.threshold_metrics(y_te, p_base, 0.5),
            # fair comparison: the baseline also gets its own cost-optimal threshold (from its OOF scores)
            "at_cost_threshold": M.threshold_metrics(y_te, p_base, M.best_threshold(y_tr, oof["logreg_raw"])),
        },
        "fairness_by_sex": M.group_report(y_te, p_te, thr, test_df[C.PROTECTED].map({1: "male", 2: "female"}).to_numpy()),
    }

    with mlflow.start_run(run_name=f"final_{best_name}_calibrated"):
        mlflow.log_param("threshold", thr)
        mlflow.log_metrics({f"test_{k}": v for k, v in M.ranking_metrics(y_te, p_te).items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test["final"]["at_cost_threshold"].items() if k != "threshold"})

    C.MODELS_DIR.mkdir(exist_ok=True)
    feature_names = list(X_fit.columns)
    joblib.dump(
        {"model": final, "threshold": thr, "engineered": best_eng, "features": feature_names, "name": best_name},
        C.MODELS_DIR / "model.joblib",
    )
    if tuned:
        joblib.dump(lgb.LGBMClassifier(**tuned).fit(X_tr_eng, y_tr), C.MODELS_DIR / "lgb_uncalibrated.joblib")

    results = {
        "dataset": {"rows": len(df), "train": len(train_df), "test": len(test_df), "default_rate": float(df[C.TARGET].mean())},
        "cost_assumption": {"cost_fn": C.COST_FN, "cost_fp": C.COST_FP},
        "cv_5fold_train": cv_results,
        "tuning": {"trials": n_trials, "seconds": round(tune_seconds, 1), "best_params": tuned},
        "threshold_chosen_on_oof": thr,
        "test": test,
    }
    C.REPORTS_DIR.mkdir(exist_ok=True)
    (C.REPORTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    np.save(C.REPORTS_DIR / "oof_scores.npy", np.vstack([oof[n] for n in cv_results]))
    print(json.dumps(test, indent=2))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    main(ap.parse_args().trials)
