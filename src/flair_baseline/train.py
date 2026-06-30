"""XGBoost training + scoring for one task (sparse features).

Fits on the train split, scores every row (train+test) so the FLAIR report can bin
by split. Demographics flow into the report via the cohort parquet, so the preds
table carries only the report-required keys + predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import xgboost as xgb

JOIN_ID = "hospitalization_join_id"

# Extra dense numeric features pulled from the cohort (appended after the count columns).
_EXTRA_FEATURES = ["age_at_admission"]

# Fixed XGBoost params used by both HPO trials and the final fit (only the
# search-space params below are tuned).
_FIXED_PARAMS = dict(tree_method="hist", random_state=0)

# Fallback params when HPO is disabled (--no-hpo) — the historical hand-tuned set.
_DEFAULT_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=1.0, reg_lambda=1.0, reg_alpha=0.0,
)


def tune_xgb(X_train: sp.csr_matrix, y_train: np.ndarray, scale_pos_weight: float,
             n_trials: int = 30, max_tune_rows: int = 2_000_000) -> dict:
    """Optuna search → best XGBoost params (maximizing 5-fold CV AUROC).

    Runs ``n_trials`` TPE-sampled trials; each scores via ``xgb.cv`` on the train
    rows only (no leakage into the report's test split). Returns the tuned param
    dict (including ``n_estimators``) — fixed params / scale_pos_weight are added
    by the caller at final-fit time.

    For very large tasks (e.g. the 14.5M-row sepsis grid) the CV would exhaust
    RAM, so the *search* runs on a deterministic ``max_tune_rows`` subsample; the
    final model is still fit on the full train split by the caller.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if X_train.shape[0] > max_tune_rows:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(X_train.shape[0], size=max_tune_rows, replace=False))
        X_tune, y_tune = X_train[idx], y_train[idx]
    else:
        X_tune, y_tune = X_train, y_train
    dtrain = xgb.DMatrix(X_tune, label=y_tune)
    # Stratified CV needs ≥ nfold samples per class; cap folds for tiny/imbalanced tasks.
    n_pos = int((y_tune == 1).sum())
    n_neg = int((y_tune == 0).sum())
    nfold = max(2, min(5, n_pos, n_neg))

    def objective(trial: "optuna.Trial") -> float:
        n_estimators = trial.suggest_int("n_estimators", 100, 600)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "scale_pos_weight": scale_pos_weight,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        cv = xgb.cv(params, dtrain, num_boost_round=n_estimators, nfold=nfold,
                    stratified=True, metrics=("auc",), seed=0)
        return float(cv["test-auc-mean"].max())

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return dict(study.best_params)


def train_and_score(X: sp.csr_matrix, prediction_ids: list[str], vocab: list[str],
                    task_df: pl.DataFrame, label_col: str, *,
                    model_out: str | None = None, vocab_out: str | None = None,
                    params_out: str | None = None, n_trials: int = 30,
                    model_in: str | None = None) -> pl.DataFrame:
    """Train (unless model_in given) and score every prediction. Returns the preds frame."""
    extras = [c for c in _EXTRA_FEATURES if c in task_df.columns]
    meta = task_df.select("prediction_id", "hospitalization_id", JOIN_ID,
                          "split", label_col, *extras)
    # Align cohort rows to X's row order (prediction_ids).
    order = pl.DataFrame({"prediction_id": prediction_ids}).with_row_index("_r")
    d = order.join(meta, on="prediction_id", how="left").sort("_r")

    if extras:
        age = d.select([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in extras]).to_numpy()
        X_full = sp.hstack([X, sp.csr_matrix(age.astype("float32"))], format="csr")
    else:
        X_full = X.tocsr()

    y = d[label_col].cast(pl.Int32).fill_null(0).to_numpy()
    split = d["split"].to_numpy()
    train_mask = split == "train"

    if model_in:
        booster = xgb.Booster()
        booster.load_model(model_in)
        prob = booster.inplace_predict(X_full)
    elif train_mask.sum() == 0 or len(np.unique(y[train_mask])) < 2 or X_full.shape[1] == 0:
        base = float(y[train_mask].mean()) if train_mask.sum() else 0.0
        prob = np.full(X_full.shape[0], base, dtype="float32")
    else:
        pos = float((y[train_mask] == 1).sum())
        neg = float((y[train_mask] == 0).sum())
        spw = neg / pos if pos > 0 else 1.0
        if n_trials and n_trials > 0:
            best = tune_xgb(X_full[train_mask], y[train_mask], spw, n_trials=n_trials)
        else:
            best = dict(_DEFAULT_PARAMS)
        clf = xgb.XGBClassifier(
            **best, **_FIXED_PARAMS, eval_metric="logloss",
            scale_pos_weight=spw, n_jobs=-1,
        )
        print(f"  fitting final model on {int(train_mask.sum()):,} rows …", flush=True)
        clf.fit(X_full[train_mask], y[train_mask])
        prob = clf.predict_proba(X_full)[:, 1]
        if model_out:
            clf.get_booster().save_model(model_out)
        if vocab_out:
            Path(vocab_out).write_text(json.dumps({"vocab": vocab, "extras": extras}))
        if params_out:
            Path(params_out).write_text(json.dumps(
                {**best, **_FIXED_PARAMS, "scale_pos_weight": spw}, indent=2))

    return d.select(
        "prediction_id", "hospitalization_id", JOIN_ID, "split", label_col,
    ).with_columns(
        pl.Series("y_prob", prob).cast(pl.Float64),
        (pl.Series("y_prob", prob) >= 0.5).cast(pl.Int32).alias("y_pred"),
    )
