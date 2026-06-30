"""Shared-MEDS building blocks: join-id membership expansion, ETL batching, fixed vocab.

These lock the two correctness-critical pieces of the shared-MEDS refactor without
needing CLIF/MIMIC access:
  * ``members_of_joins`` resolves a cohort's join_ids back to ALL member
    hospitalizations (the sibling-inclusion fix), with a self-fallback for
    never-stitched encounters;
  * ``_batches`` partitions the hosp-id list for the ``-pmc`` batched ETL exactly,
    with no dropped or duplicated ids;
  * a fixed (supplied) vocabulary yields the SAME column space across different
    cohorts — the invariant that makes shipped model bundles interchangeable.
"""
from __future__ import annotations

from datetime import datetime

import polars as pl

from flair_benchmark._stitch import members_of_joins
from flair_benchmark.features.fe_meds import _batches
from flair_baseline.featurize import count_features


def _idx() -> pl.DataFrame:
    # h1 & h2 stitch into encounter J1; h3 is its own block J3 (present in index).
    # h9 is never stitched (absent from the index) → self-mapped at attach time.
    return pl.DataFrame({
        "hospitalization_id": ["h1", "h2", "h3"],
        "hospitalization_join_id": ["J1", "J1", "J3"],
        "patient_id": ["p1", "p1", "p3"],
    })


def test_members_of_joins_expands_siblings():
    """A cohort holding only h1 must pull its stitched sibling h2 via encounter J1."""
    got = members_of_joins(_idx(), ["J1"])
    assert got == ["h1", "h2"]   # h2 recovered even though the cohort never named it


def test_members_of_joins_self_fallback():
    """A never-stitched join_id (absent from the index) maps to its own id."""
    got = members_of_joins(_idx(), ["J1", "h9"])
    assert set(got) == {"h1", "h2", "h9"}   # h9 self-covered, J1 fully expanded


def test_members_of_joins_no_synthetic_leak():
    """Only matched member ids + unmatched self-ids — no stray join labels leak in."""
    got = members_of_joins(_idx(), ["J3"])
    assert got == ["h3"]   # J3's only member, and 'J3' itself is not a hosp id


def test_batches_partition_exactly():
    items = list(range(10))
    assert _batches(items, 3) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    # union of batches == input, order preserved, nothing dropped/duplicated.
    flat = [x for b in _batches(items, 4) for x in b]
    assert flat == items
    assert _batches([], 5) == []
    assert _batches([1, 2], 5) == [[1, 2]]   # batch larger than input → one batch


# --- fixed vocabulary gives a constant column space across cohorts --------------
_EVENTS_A = pl.DataFrame({
    "hospitalization_join_id": ["A", "A"],
    "time": [datetime(2020, 1, 1, 6), datetime(2020, 1, 1, 8)],
    "code": ["LAB//lactate//mmol/l", "RESP//device_category//imv"],
    "numeric_value": [1.0, None],
})
_COHORT_A = pl.DataFrame({
    "prediction_id": ["pA"], "hospitalization_join_id": ["A"],
    "prediction_dttm": [datetime(2020, 1, 2)], "split": ["train"], "label": [1],
})
_EVENTS_B = pl.DataFrame({
    "hospitalization_join_id": ["B"],
    "time": [datetime(2020, 1, 1, 7)],
    "code": ["LAB//lactate//mmol/l"],          # B never sees imv at all
    "numeric_value": [1.1],
})
_COHORT_B = pl.DataFrame({
    "prediction_id": ["pB"], "hospitalization_join_id": ["B"],
    "prediction_dttm": [datetime(2020, 1, 2)], "split": ["train"], "label": [0],
})

_FIXED_VOCAB = ["LAB//lactate", "RESP//device_category//imv", "MED_INT//ghostdrug"]


def test_fixed_vocab_constant_width_across_cohorts():
    """The same supplied vocab → identical column count for two unrelated cohorts."""
    Xa, _, va = count_features(_EVENTS_A, _COHORT_A, "label", vocab=_FIXED_VOCAB)
    Xb, _, vb = count_features(_EVENTS_B, _COHORT_B, "label", vocab=_FIXED_VOCAB)
    assert va == vb == _FIXED_VOCAB
    assert Xa.shape[1] == Xb.shape[1] == len(_FIXED_VOCAB)
    # A code absent from a site is an all-zero column, never a stored 0.
    ghost = _FIXED_VOCAB.index("MED_INT//ghostdrug")
    assert Xa[:, ghost].nnz == 0 and Xb[:, ghost].nnz == 0
    assert (Xb.data == 0).sum() == 0
