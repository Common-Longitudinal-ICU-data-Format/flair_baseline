"""Model-kind routing and transfer learning (synthetic data — no real/MIMIC access).

Locks the contract that lets one site hold three answers at once:
  * per-model artifacts (preds, report, model bundle) carry a kind segment;
    per-data artifacts (cohort, features, codes, table1) do NOT;
  * a shipped bundle resolves whether it was built before or after the split;
  * transfer learning genuinely continues the base ensemble rather than
    replacing it, and refuses a base model whose column space disagrees.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
import xgboost as xgb
from flair_baseline.layout import (
    KIND_EXTERNAL,
    KIND_LOCAL,
    KIND_TRANSFER,
    KINDS,
    TaskPaths,
    resolve_shipped_model,
)
from flair_baseline.train import train_and_score, tune_xgb

TASK = "extubation_failure_24h"
LABEL = "label"


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
def test_per_model_artifacts_carry_the_kind(tmp_path):
    p = TaskPaths.make(tmp_path, "siteb", TASK)
    for kind in KINDS:
        assert p.preds(kind).name == f"preds_{kind}.parquet"
        assert p.report_dir(kind).parent.name == "report"
        assert p.report_dir(kind).name == kind
        assert p.model(kind).parent.name == kind
        assert p.vocab(kind).parent.name == kind
        assert p.params(kind).parent.name == kind
    # The three kinds never collide.
    assert len({str(p.preds(k)) for k in KINDS}) == len(KINDS)
    assert len({str(p.model(k)) for k in KINDS}) == len(KINDS)


def test_per_data_artifacts_have_no_kind(tmp_path):
    """cohort/features/codes/table1 describe the site's DATA, so they are shared
    across kinds — duplicating them per kind is the bug this guards."""
    p = TaskPaths.make(tmp_path, "siteb", TASK)
    for path in (p.cohort, p.features, p.codes, p.table1):
        assert not any(k in path.parts for k in KINDS), path


def test_unknown_kind_is_rejected(tmp_path):
    p = TaskPaths.make(tmp_path, "siteb", TASK)
    with pytest.raises(ValueError, match="unknown model kind"):
        p.report_dir("locale")


def test_mkdirs_creates_only_the_requested_kind(tmp_path):
    p = TaskPaths.make(tmp_path, "siteb", TASK)
    p.mkdirs(KIND_LOCAL)
    assert p.report_dir(KIND_LOCAL).is_dir()
    assert p.model_dir(KIND_LOCAL).is_dir()
    assert not p.report_dir(KIND_TRANSFER).exists()
    assert not p.model_dir(KIND_EXTERNAL).exists()


# --------------------------------------------------------------------------- #
# shipped-bundle resolution (back-compat)
# --------------------------------------------------------------------------- #
def _write_bundle(d):
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.json").write_text("{}")
    (d / "vocab.json").write_text(json.dumps({"vocab": ["LAB//lactate"]}))


def test_resolves_new_nested_bundle(tmp_path):
    _write_bundle(tmp_path / TASK / KIND_LOCAL)
    model, vocab = resolve_shipped_model(tmp_path, TASK)
    assert model.parent.name == KIND_LOCAL and vocab.parent.name == KIND_LOCAL


def test_resolves_pre_kind_flat_bundle(tmp_path):
    """Bundles shipped before the kind split live flat at <task>/."""
    _write_bundle(tmp_path / TASK)
    model, vocab = resolve_shipped_model(tmp_path, TASK)
    assert model.parent.name == TASK


def test_nested_bundle_wins_over_flat(tmp_path):
    _write_bundle(tmp_path / TASK)
    _write_bundle(tmp_path / TASK / KIND_LOCAL)
    model, _ = resolve_shipped_model(tmp_path, TASK)
    assert model.parent.name == KIND_LOCAL


def test_missing_or_half_bundle_resolves_to_none(tmp_path):
    assert resolve_shipped_model(tmp_path, TASK) is None
    (tmp_path / TASK).mkdir(parents=True)
    (tmp_path / TASK / "model.json").write_text("{}")   # vocab.json absent
    assert resolve_shipped_model(tmp_path, TASK) is None


# --------------------------------------------------------------------------- #
# transfer learning
# --------------------------------------------------------------------------- #
def _synthetic(n=400, n_feat=6, seed=0):
    """A learnable signal: label follows the sign of a linear combination."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat)).astype("float32")
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    cohort = pl.DataFrame({
        "prediction_id": [f"p{i}" for i in range(n)],
        "hospitalization_id": [f"h{i}" for i in range(n)],
        "hospitalization_join_id": [f"j{i}" for i in range(n)],
        "split": ["train"] * (n // 2) + ["test"] * (n - n // 2),
        LABEL: y.tolist(),
    })
    return X, cohort


def _base_model(tmp_path, X, cohort, n_trees=10):
    clf = xgb.XGBClassifier(n_estimators=n_trees, max_depth=3, tree_method="hist",
                            random_state=0)
    train = (cohort["split"] == "train").to_numpy()
    clf.fit(X[train], cohort[LABEL].to_numpy()[train])
    out = tmp_path / "base.json"
    clf.get_booster().save_model(out)
    return out


def test_transfer_appends_trees_to_the_base_ensemble(tmp_path):
    X, cohort = _synthetic()
    base = _base_model(tmp_path, X, cohort, n_trees=10)
    names = [f"f{i}" for i in range(X.shape[1])]
    out = tmp_path / "transferred.json"

    preds = train_and_score(X, cohort["prediction_id"].to_list(), names, cohort, LABEL,
                            base_model=str(base), model_out=str(out), n_trials=0)

    grown = xgb.Booster()
    grown.load_model(out)
    assert grown.num_boosted_rounds() > 10, "transfer must CONTINUE the base ensemble"
    assert preds.height == X.shape[0]
    # Scoring the transferred model is self-contained — no companion file needed.
    assert not np.allclose(preds["y_prob"].to_numpy(),
                           xgb.Booster(model_file=str(base)).inplace_predict(X))


def test_transfer_rejects_a_base_model_of_the_wrong_width(tmp_path):
    X, cohort = _synthetic(n_feat=6)
    base = _base_model(tmp_path, X, cohort)
    X_narrow = X[:, :4]          # local features disagree with the shipped vocab
    names = [f"f{i}" for i in range(4)]
    with pytest.raises(ValueError, match="features"):
        train_and_score(X_narrow, cohort["prediction_id"].to_list(), names, cohort,
                        LABEL, base_model=str(base), n_trials=0)


def test_transfer_falls_back_to_the_base_when_nothing_is_fittable(tmp_path):
    """A degenerate train split must not throw the transfer away for a constant."""
    X, cohort = _synthetic()
    base = _base_model(tmp_path, X, cohort)
    single_class = cohort.with_columns(pl.when(pl.col("split") == "train")
                                       .then(0).otherwise(pl.col(LABEL)).alias(LABEL))
    names = [f"f{i}" for i in range(X.shape[1])]
    preds = train_and_score(X, cohort["prediction_id"].to_list(), names, single_class,
                            LABEL, base_model=str(base), n_trials=0)
    assert preds["y_prob"].n_unique() > 1, "should score the base model, not a constant"


def test_tune_xgb_accepts_a_base_margin(tmp_path):
    """xgb.cv takes no xgb_model=, so transfer HPO rides on base_margin instead."""
    X, cohort = _synthetic()
    train = (cohort["split"] == "train").to_numpy()
    y = cohort[LABEL].to_numpy()[train]
    base = xgb.Booster(model_file=str(_base_model(tmp_path, X, cohort)))
    margin = base.inplace_predict(X[train], predict_type="margin")

    best = tune_xgb(X[train], y, 1.0, n_trials=2, base_margin=margin)
    assert {"n_estimators", "max_depth", "learning_rate"} <= set(best)
