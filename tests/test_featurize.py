"""Featurizer role/missingness invariants (synthetic data — no real/MIMIC access).

Locks the contract that replaced count-only featurization:
  * ROLE_STAT codes emit min/max/mean/median over numeric_value, and a code that was
    never measured is NaN — never 0, which would read as a real measurement;
  * ROLE_COUNT (medications) emits an event count whose value ignores the dose, and
    "never given" is a genuine 0;
  * ROLE_ONEHOT (RESP device/mode) emits 0/1 presence, never a count;
  * the point-in-time filter is strict — an event at or after prediction_dttm is
    invisible to every statistic.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
from flair_baseline.featurize import (
    ROLE_COUNT,
    ROLE_ONEHOT,
    ROLE_STAT,
    STATS,
    code_role,
    count_features,
    feature_names,
)

# Two encounters, prediction at 2020-01-02. Events strictly before are aggregated.
# A: 3 lactates + 2 imv + 2 vancomycin before, 1 lactate after (must be dropped).
# B: 1 lactate before, and NO imv / NO vancomycin at all.
_EVENTS = pl.DataFrame(
    {
        "hospitalization_join_id": ["A", "A", "A", "A", "A", "A", "A", "A", "B"],
        "time": [
            datetime(2020, 1, 1, 6),   # A lactate 1.0  (before)
            datetime(2020, 1, 1, 8),   # A lactate 1.2  (before)
            datetime(2020, 1, 1, 10),  # A lactate 2.8  (before)
            datetime(2020, 1, 3, 0),   # A lactate 99.0 (AFTER prediction — excluded)
            datetime(2020, 1, 1, 9),   # A imv          (before)
            datetime(2020, 1, 1, 11),  # A imv          (before, 2nd → still 1.0)
            datetime(2020, 1, 1, 7),   # A vancomycin dose 1000
            datetime(2020, 1, 1, 19),  # A vancomycin dose 1250
            datetime(2020, 1, 1, 7),   # B lactate 1.1  (before)
        ],
        "code": [
            "LAB//lactate//mmol/l",
            "LAB//lactate//mmol/l",
            "LAB//lactate//mmol/l",
            "LAB//lactate//mmol/l",
            "RESP//device_category//imv",
            "RESP//device_category//imv",
            "MED_INT//vancomycin//mg//given",
            "MED_INT//vancomycin//mg//given",
            "LAB//lactate//mmol/l",
        ],
        "numeric_value": [1.0, 1.2, 2.8, 99.0, None, None, 1000.0, 1250.0, 1.1],
    }
)

_COHORT = pl.DataFrame(
    {
        "prediction_id": ["pA", "pB"],
        "hospitalization_join_id": ["A", "B"],
        "prediction_dttm": [datetime(2020, 1, 2), datetime(2020, 1, 2)],
        "split": ["train", "test"],
        "label": [1, 0],
    }
)

# truncated codes: numeric→2 levels, categorical→3 levels
_LACTATE = "LAB//lactate"
_IMV = "RESP//device_category//imv"
_VANC = "MED_INT//vancomycin"


def _featurize(**kw):
    """Run the featurizer and return (X, {feature_name: column index}, row index)."""
    X, ids, names = count_features(_EVENTS, _COHORT, "label", **kw)
    return X, {n: i for i, n in enumerate(names)}, {p: i for i, p in enumerate(ids)}


def test_roles_are_assigned_by_domain():
    assert code_role(_LACTATE) == ROLE_STAT
    assert code_role("VITAL//heart_rate") == ROLE_STAT
    assert code_role("PA//rass") == ROLE_STAT
    assert code_role("RESP//peep_set") == ROLE_STAT
    assert code_role("RESP//tracheostomy") == ROLE_STAT
    assert code_role(_VANC) == ROLE_COUNT
    assert code_role("MED_CON//norepinephrine") == ROLE_COUNT
    assert code_role(_IMV) == ROLE_ONEHOT
    assert code_role("RESP//mode_category//pressure_support/cpap") == ROLE_ONEHOT


def test_feature_names_expand_stat_codes_only():
    names = feature_names([_LACTATE, _VANC, _IMV])
    assert names == [f"{_LACTATE}::{s}" for s in STATS] + [_VANC, _IMV]


def test_stat_columns_summarize_values_before_prediction():
    """min/max/mean/median over the pre-prediction values; the later 99.0 is invisible."""
    X, col, row = _featurize()
    a = row["pA"]
    # A's pre-prediction lactates are 1.0, 1.2, 2.8 (99.0 is at 2020-01-03).
    assert X[a, col[f"{_LACTATE}::min"]] == np.float32(1.0)
    assert X[a, col[f"{_LACTATE}::max"]] == np.float32(2.8)
    assert X[a, col[f"{_LACTATE}::mean"]] == np.float32((1.0 + 1.2 + 2.8) / 3)
    assert X[a, col[f"{_LACTATE}::median"]] == np.float32(1.2)
    # B drew a single lactate → all four statistics collapse to that value.
    b = row["pB"]
    for s in STATS:
        assert X[b, col[f"{_LACTATE}::{s}"]] == np.float32(1.1)


def test_unmeasured_stat_is_nan_not_zero():
    """A ROLE_STAT code with no events must be missing, not a measurement of 0."""
    vocab = [_LACTATE, "LAB//troponin_t", _VANC, _IMV]
    X, col, _row = _featurize(vocab=vocab)
    for s in STATS:
        assert np.isnan(X[:, col[f"LAB//troponin_t::{s}"]]).all()
    # ...and the observed lactate columns are emphatically not NaN.
    assert not np.isnan(X[:, col[f"{_LACTATE}::min"]]).any()


def test_med_count_ignores_dose_and_zero_means_never_given():
    X, col, row = _featurize()
    # A got two vancomycin doses (1000 and 1250 mg) → count 2, not the dose.
    assert X[row["pA"], col[_VANC]] == np.float32(2.0)
    # B never got it → genuine 0, distinguishable from a missing lab.
    assert X[row["pB"], col[_VANC]] == np.float32(0.0)
    assert not np.isnan(X[:, col[_VANC]]).any()


def test_onehot_is_presence_not_count():
    X, col, row = _featurize()
    # A has TWO imv records but the column is a presence indicator.
    assert X[row["pA"], col[_IMV]] == np.float32(1.0)
    assert X[row["pB"], col[_IMV]] == np.float32(0.0)


def test_missing_vocab_code_never_invents_data():
    """A supplied vocab code absent from the site → NaN stats / 0 counts, never junk."""
    vocab = [_LACTATE, _IMV, "MED_INT//ghostdrug", "LAB//ghostlab"]
    X, col, _row = _featurize(vocab=vocab)
    assert X.shape[1] == len(feature_names(vocab))
    assert (X[:, col["MED_INT//ghostdrug"]] == 0.0).all()
    for s in STATS:
        assert np.isnan(X[:, col[f"LAB//ghostlab::{s}"]]).all()


def test_matrix_width_is_fixed_by_vocab_not_by_site_data():
    """Same vocab ⇒ same width, whichever codes this site happens to have."""
    vocab = [_LACTATE, "LAB//ghostlab", _VANC, _IMV]
    expected = len(feature_names(vocab))
    X_all, _col, _row = _featurize(vocab=vocab)
    # Encounter B alone has no imv and no vancomycin at all.
    X_one, _ids, _names = count_features(
        _EVENTS.filter(pl.col("hospitalization_join_id") == "B"),
        _COHORT.filter(pl.col("prediction_id") == "pB"), "label", vocab=vocab)
    assert X_all.shape[1] == expected
    assert X_one.shape[1] == expected
