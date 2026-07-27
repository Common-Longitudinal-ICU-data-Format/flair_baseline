"""XGBoost training + scoring for one task (dense float32 features).

Fits on the train split, scores every row (train+test) so the FLAIR report can bin
by split. Demographics flow into the report via the cohort parquet, so the preds
table carries only the report-required keys + predictions.

NaN is load-bearing here, not incidental: the featurizer emits NaN for a statistic
that was never measured (as distinct from a genuine 0 count), and XGBoost routes
those cells down its own missing-value branch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

JOIN_ID = "hospitalization_join_id"

# Dense extra features appended after the featurizer's columns, in this fixed order:
#   [ _EXTRA_NUMERIC... , sex__<cat>... , time_since_first_hrs ]
# The order is fixed in code (not read back from the sidecar) so train and infer —
# and every site — produce an identical column layout.
_EXTRA_NUMERIC = ["age_at_admission"]     # continuous cohort numerics (fill_null → 0)
# Fixed one-hot sex columns. A present sex → 1.0/0.0; a sex outside this set
# (null / Unknown / Other) → NaN in BOTH columns so XGBoost routes it as missing.
_SEX_CATEGORIES = ["Female", "Male"]


def _build_extras(d: pl.DataFrame, time_since_first: np.ndarray | None
                  ) -> tuple[np.ndarray | None, list[str]]:
    """Assemble dense extra feature columns + their names, in the fixed order.

    ``d`` is the cohort rows already aligned to the CSR row order; ``time_since_first``
    (if given) is in that same row order. Returns (dense matrix or None, names).
    """
    cols: list[np.ndarray] = []
    names: list[str] = []
    for c in _EXTRA_NUMERIC:
        if c in d.columns:
            cols.append(d[c].cast(pl.Float64).fill_null(0.0).to_numpy())
            names.append(c)
    if "sex_category" in d.columns:
        for cat in _SEX_CATEGORIES:
            # present rows → 1.0/0.0; sex ∉ _SEX_CATEGORIES → null → NaN (missing).
            col = (
                pl.when(pl.col("sex_category").is_in(_SEX_CATEGORIES))
                .then((pl.col("sex_category") == cat).cast(pl.Float64))
                .otherwise(None)
            )
            cols.append(d.select(col.alias("s"))["s"].to_numpy())
            names.append(f"sex__{cat}")
    if time_since_first is not None:
        cols.append(np.asarray(time_since_first, dtype="float64"))
        names.append("time_since_first_hrs")
    if not cols:
        return None, []
    return np.column_stack(cols), names

# Fixed XGBoost params used by both HPO trials and the final fit (only the
# search-space params below are tuned).
_FIXED_PARAMS = dict(tree_method="hist", random_state=0)

# Fallback params when HPO is disabled (--no-hpo) — the historical hand-tuned set.
_DEFAULT_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=1.0, reg_lambda=1.0, reg_alpha=0.0,
)


def tune_xgb(X_train: np.ndarray, y_train: np.ndarray, scale_pos_weight: float,
             n_trials: int = 30, max_tune_rows: int = 2_000_000,
             base_margin: np.ndarray | None = None) -> dict:
    """Optuna search → best XGBoost params (maximizing 5-fold CV AUROC).

    Runs ``n_trials`` TPE-sampled trials; each scores via ``xgb.cv`` on the train
    rows only (no leakage into the report's test split). Returns the tuned param
    dict (including ``n_estimators``) — fixed params / scale_pos_weight are added
    by the caller at final-fit time.

    For very large tasks (a multi-million-row hourly grid) the CV would exhaust
    RAM, so the *search* runs on a deterministic ``max_tune_rows`` subsample; the
    final model is still fit on the full train split by the caller.

    ``base_margin`` is how transfer learning gets tuned. ``xgb.cv`` takes no
    ``xgb_model=`` argument, so continued boosting cannot be cross-validated
    directly — but boosting on top of an existing ensemble is exactly boosting
    with that ensemble's raw output as a per-row offset, which ``xgb.cv`` does
    support. ``DMatrix.slice`` carries the margin into each fold, so the search
    space and trial count stay identical to a from-scratch fit.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if X_train.shape[0] > max_tune_rows:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(X_train.shape[0], size=max_tune_rows, replace=False))
        X_tune, y_tune = X_train[idx], y_train[idx]
        margin_tune = None if base_margin is None else base_margin[idx]
    else:
        X_tune, y_tune = X_train, y_train
        margin_tune = base_margin
    dtrain = xgb.DMatrix(X_tune, label=y_tune)
    if margin_tune is not None:
        dtrain.set_base_margin(np.asarray(margin_tune, dtype="float32"))
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


def train_and_score(X: np.ndarray, prediction_ids: list[str],
                    feature_names: list[str], task_df: pl.DataFrame, label_col: str, *,
                    model_out: str | None = None, vocab_out: str | None = None,
                    vocab_meta: dict | None = None,
                    params_out: str | None = None, n_trials: int = 30,
                    model_in: str | None = None, base_model: str | None = None,
                    time_since_first: np.ndarray | None = None) -> pl.DataFrame:
    """Train (unless model_in given) and score every prediction. Returns the preds frame.

    ``model_in`` scores a frozen model and fits nothing. ``base_model`` does the
    opposite: it seeds a fit, appending locally-trained trees to a shipped
    ensemble. The saved model is self-contained (base trees + local trees), so
    scoring it later needs no companion file.
    """
    have = [c for c in (*_EXTRA_NUMERIC, "sex_category") if c in task_df.columns]
    meta = task_df.select("prediction_id", "hospitalization_id", JOIN_ID,
                          "split", label_col, *have)
    # Align cohort rows to X's row order (prediction_ids); time_since_first is
    # already in that order, so it aligns to d/X positionally.
    order = pl.DataFrame({"prediction_id": prediction_ids}).with_row_index("_r")
    d = order.join(meta, on="prediction_id", how="left").sort("_r")

    extra_mat, extras = _build_extras(d, time_since_first)
    X = np.asarray(X, dtype="float32")
    if extra_mat is not None:
        X_full = np.hstack([X, extra_mat.astype("float32")])
    else:
        X_full = X

    y = d[label_col].cast(pl.Int32).fill_null(0).to_numpy()
    split = d["split"].to_numpy()
    train_mask = split == "train"

    base_booster = None
    if base_model:
        base_booster = xgb.Booster()
        base_booster.load_model(base_model)
        # The base trees index into the SOURCE site's column layout. A width
        # mismatch means the shipped vocab and the local features disagree, and
        # every appended tree would be split on the wrong feature.
        n_base = base_booster.num_features()
        if n_base != X_full.shape[1]:
            raise ValueError(
                f"base model expects {n_base:,} features but the local matrix has "
                f"{X_full.shape[1]:,}. The shipped vocab.json and this site's "
                f"features.npz disagree — re-featurize against the shipped vocabulary.")

    if model_in:
        booster = xgb.Booster()
        booster.load_model(model_in)
        prob = booster.inplace_predict(X_full)
    elif train_mask.sum() == 0 or len(np.unique(y[train_mask])) < 2 or X_full.shape[1] == 0:
        # Nothing fittable. With a base model in hand, scoring it beats emitting a
        # constant — falling back to the prevalence would throw away the transfer.
        if base_booster is not None:
            prob = base_booster.inplace_predict(X_full)
        else:
            base = float(y[train_mask].mean()) if train_mask.sum() else 0.0
            prob = np.full(X_full.shape[0], base, dtype="float32")
    else:
        # scale_pos_weight removed (was neg/pos). Upweighting the rare positive class
        # improves ranking/recall but inflates predicted risk (O:E > 1) and wrecks
        # probability calibration. Neutral weight keeps predictions on the prevalence
        # scale; class imbalance is instead reflected honestly in the probabilities.
        spw = 1.0
        margin = (None if base_booster is None else
                  base_booster.inplace_predict(X_full[train_mask], predict_type="margin"))
        if n_trials and n_trials > 0:
            best = tune_xgb(X_full[train_mask], y[train_mask], spw, n_trials=n_trials,
                            base_margin=margin)
        else:
            best = dict(_DEFAULT_PARAMS)
        clf = xgb.XGBClassifier(
            **best, **_FIXED_PARAMS, eval_metric="logloss",
            scale_pos_weight=spw, n_jobs=-1, missing=np.nan,
        )
        seeded = "" if base_booster is None else f" on top of {base_booster.num_boosted_rounds():,} base trees"
        print(f"  fitting final model on {int(train_mask.sum()):,} rows{seeded} …", flush=True)
        clf.fit(X_full[train_mask], y[train_mask], xgb_model=base_booster)
        prob = clf.predict_proba(X_full)[:, 1]
        if model_out:
            clf.get_booster().save_model(model_out)
        if vocab_out:
            # "vocab"/"roles" are the truncated-code space (what infer resolves
            # against); "features" is the expanded column layout they produce.
            Path(vocab_out).write_text(json.dumps(
                {**(vocab_meta or {}), "features": feature_names, "extras": extras}))
        if params_out:
            Path(params_out).write_text(json.dumps(
                {**best, **_FIXED_PARAMS, "scale_pos_weight": spw}, indent=2))

    return d.select(
        "prediction_id", "hospitalization_id", JOIN_ID, "split", label_col,
    ).with_columns(
        pl.Series("y_prob", prob).cast(pl.Float64),
        (pl.Series("y_prob", prob) >= 0.5).cast(pl.Int32).alias("y_pred"),
    )
