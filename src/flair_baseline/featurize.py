"""Leak-free point-in-time featurizer (integerized, block-chunked, role-aware).

Every feature summarizes the events of one (level-truncated) code in the same encounter
(``hospitalization_join_id``) with ``time < prediction_dttm`` — strict point-in-time, no
leakage. *How* they are summarized depends on the code's **role**:

* ``ROLE_STAT``   → four columns ``code::min|max|mean|median`` over ``numeric_value``.
  Labs, vitals, GCS/RASS, and the respiratory-support parameters. Never measured ⇒
  **NaN** (not 0) so XGBoost routes the cell as missing.
* ``ROLE_COUNT``  → one column, #events. Medications, where the dose magnitude is far
  less informative than the fact and frequency of exposure. Never given ⇒ genuine 0.
* ``ROLE_ONEHOT`` → one column, 1 if the code ever occurred before now else 0.
  ``RESP//device_category//*`` and ``RESP//mode_category//*``. Not mutually exclusive
  across a history window — a patient escalating nasal_cannula → imv is 1 in both.

Feature granularity — ELF codes are ``//``-delimited hierarchies whose raw depth
varies (``LAB//lactate//mmol/l//bmp`` is 4 levels, ``RESP//device_category//imv`` is
3). Before aggregating, each code is truncated to a fixed depth by value type so that
the **unit is never a feature**:

* **numeric** codes (any event carries a ``numeric_value``) → **2 levels**
  ``DOMAIN//concept``  (``LAB//lactate``, ``MED_INT//vancomycin``, ``VITAL//heart_rate``)
* **categorical** codes (text-only) → **3 levels**
  ``DOMAIN//category//value``  (``RESP//device_category//imv``)

So every lactate draw feeds the same four columns regardless of unit (mmol/l vs mg/dl)
or order type, and every vancomycin dose the same counter regardless of unit/action.
The numeric/categorical split mirrors ``is_numeric_value`` in ``metadata/codes.parquet``.

Scale: the event→prediction join is integerized (truncated codes and encounter blocks
are factorized to uint32 first, so the ~100M-row join never carries strings), the event
side is materialized once, and the join then runs in ``n_chunks`` passes partitioned by
a block-hash so the *exploded* event×prediction frame — the actual blowup — peaks at
~1/n_chunks. The result is a dense float32 matrix: at 442 columns even a 262k-row
cohort is under 500 MB, and dense suits ``tree_method="hist"`` better than a CSR that
would have to store every NaN explicitly.

Vocabulary is fit on the train split only, or supplied via ``vocab=`` (scoring a new
site against a saved model). A saved vocab is in the same truncated space.
"""
from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm

import numpy as np
import polars as pl

JOIN_ID = "hospitalization_join_id"

# Truncation depth by value type: numeric ⇒ DOMAIN//concept, categorical ⇒
# DOMAIN//category//value. Dropping deeper levels removes the unit (and order/action)
# from the feature space.
NUMERIC_LEVELS = 2
CATEGORICAL_LEVELS = 3

# How a code's events become feature columns. See the module docstring.
ROLE_STAT, ROLE_COUNT, ROLE_ONEHOT = 0, 1, 2

# The four statistics emitted per ROLE_STAT code, in column order. Changing this
# tuple changes the matrix width and invalidates every cached features.npz — bump
# CACHE_FORMAT below if you do.
STATS = ("min", "max", "mean", "median")

# Domains whose events are counted rather than summarized. Medications carry a
# numeric_value (the dose), but exposure and frequency carry the signal — and MIMIC's
# mar_action_category values are all real administration events (given / dose_change /
# start / stop / going; no held or cancelled), so counting every event is sound.
COUNT_PREFIXES = ("MED_CON//", "MED_INT//")
# Domains whose truncated code already *is* the category value → presence indicator.
ONEHOT_PREFIXES = ("RESP//device_category//", "RESP//mode_category//")

# On-disk layout of features.npz + its sidecar. Bump on ANY change to what the cache
# stores; load_counts refuses a mismatch rather than silently reading a stale shape.
#   1 — count-only COO triplet
#   2 — + per-role statistics block (srr/scc/svals) and role_by_cint
CACHE_FORMAT = 2


def code_role(trunc: str) -> int:
    """Role of a *truncated* code — a pure function of the string, by design.

    Truncation depth already encodes the numeric/categorical decision made when the
    vocabulary was built (numeric ⇒ 2 levels, categorical ⇒ 3), so the role can be
    recovered from a committed ``vocab.json`` without re-reading any site's data. That
    keeps the column layout identical between train and infer even if a new site
    happens to classify a code's numericness differently — such a code would truncate
    to a different string and simply not match the fixed vocabulary.
    """
    if trunc.startswith(COUNT_PREFIXES):
        return ROLE_COUNT
    if trunc.startswith(ONEHOT_PREFIXES):
        return ROLE_ONEHOT
    # 2 levels ⇒ the code was numeric at build time ⇒ summarize its values.
    return ROLE_STAT if trunc.count("//") == 1 else ROLE_ONEHOT


def feature_names(vocab: list[str], roles: list[int] | None = None) -> list[str]:
    """Expand a truncated-code vocabulary into the final column names, in order."""
    roles = roles if roles is not None else [code_role(c) for c in vocab]
    names: list[str] = []
    for code, role in zip(vocab, roles):
        if role == ROLE_STAT:
            names.extend(f"{code}::{s}" for s in STATS)
        else:
            names.append(code)
    return names


def truncated_code_map(ev: pl.LazyFrame) -> pl.DataFrame:
    """Map every raw code → (truncated code, ``cint`` feature id, ``role``).

    A code is *numeric* if any of its events has a non-null ``numeric_value`` (the same
    rule ``codes.parquet`` uses for ``is_numeric_value``); numeric codes keep the first
    ``NUMERIC_LEVELS`` ``//`` levels, the rest keep ``CATEGORICAL_LEVELS``. The truncated
    codes are factorized to a contiguous ``cint`` so the heavy join stays integerized and
    raw codes that collapse to the same feature share a ``cint`` (their events merge).

    Returns one row per raw code: ``code`` (raw), ``trunc``, ``cint``, ``role``.
    """
    code_meta = (
        ev.select("code", "numeric_value")
        .group_by("code")
        .agg(pl.col("numeric_value").is_not_null().any().alias("is_numeric"))
        .collect(engine="streaming")
    )
    parts = pl.col("code").str.split("//")
    trunc = (
        pl.when(pl.col("is_numeric"))
        .then(parts.list.slice(0, NUMERIC_LEVELS).list.join("//"))
        .otherwise(parts.list.slice(0, CATEGORICAL_LEVELS).list.join("//"))
        .alias("trunc")
    )
    code_meta = code_meta.with_columns(trunc)
    trunc_ids = (
        code_meta.select("trunc").unique().sort("trunc").with_row_index("cint")
        .with_columns(
            pl.col("cint").cast(pl.UInt32),
            pl.col("trunc").map_elements(code_role, return_dtype=pl.UInt8).alias("role"),
        )
    )
    return code_meta.join(trunc_ids, on="trunc").select("code", "trunc", "cint", "role")


@dataclass
class Counts:
    """The vocab-independent output of the point-in-time aggregation join.

    Two COO blocks over the same *truncated-code* space (``cc``/``scc`` index
    ``trunc_by_cint``), because the two roles carry different payloads:

    * ``(rr, cc, nn)``      — event counts, for ROLE_COUNT and ROLE_ONEHOT codes.
      Stored raw; the 0/1 clamp for ROLE_ONEHOT happens at matrix-build time so the
      cache stays a faithful record and survives a change to that rule.
    * ``(srr, scc, svals)`` — ROLE_STAT codes; ``svals`` is ``(nnz, len(STATS))``
      float32 with columns in ``STATS`` order.

    ``split`` is per-row so vocab can be fit on the train split at ``counts_to_X``
    time. This is the expensive part to compute, so it is what gets cached to disk
    (see ``save_counts``).
    """
    rr: np.ndarray            # row index (prediction row r)
    cc: np.ndarray            # truncated-code id (cint)
    nn: np.ndarray            # count (float32)
    trunc_by_cint: list[str]  # cint → truncated code name
    prediction_ids: list[str]  # row order
    split: np.ndarray         # per-row split label ("train"/"test")
    n_rows: int
    # Per-row hours from the encounter's first MEDS event to the prediction time
    # (prediction_dttm − min(event.time) for that join_id). A dense extra feature,
    # aligned to prediction_ids; carried here so it survives the counts cache.
    time_since_first: np.ndarray = None  # float32[n_rows]
    # ROLE_STAT block (see above). Defaulted so an in-memory Counts built by older
    # callers still constructs; a *cached* one is rejected by CACHE_FORMAT instead.
    srr: np.ndarray = None            # uint32[nnz_stat]
    scc: np.ndarray = None            # uint32[nnz_stat]
    svals: np.ndarray = None          # float32[nnz_stat, len(STATS)]
    role_by_cint: list[int] = None    # cint → ROLE_*


def compute_counts(events, task_df: pl.DataFrame, n_chunks: int = 8,
                   desc: str | None = None) -> Counts:
    """Run the heavy, vocab-independent point-in-time aggregation → ``Counts``.

    This is everything up to (but not including) vocabulary resolution, so the
    result is reusable for both training (vocab fit on train) and inference
    (vocab supplied) over the same cohort. Cache it with ``save_counts``.
    """
    ev = events if isinstance(events, pl.LazyFrame) else events.lazy()

    preds = (
        task_df.select("prediction_id", JOIN_ID,
                       pl.col("prediction_dttm").cast(pl.Datetime("us")), "split")
        .with_row_index("r")
    )
    prediction_ids = preds["prediction_id"].to_list()
    split = preds["split"].to_numpy()
    n_rows = preds.height

    # Factorize blocks (+ a hash chunk) to uint32/uint8 so the join stays lean.
    blocks = (
        preds.select(JOIN_ID).unique()
        .with_row_index("jint")
        .with_columns(
            pl.col("jint").cast(pl.UInt32),
            (pl.col("jint") % n_chunks).cast(pl.UInt8).alias("chunk"),
        )
    )
    # Truncate codes by value type (numeric ⇒ 2 levels, categorical ⇒ 3) and factorize
    # the truncated codes → cint. Raw codes that share a truncated code share a cint, so
    # the per-(r, cint) aggregation below merges their events (e.g. lactate in any unit).
    code_map = truncated_code_map(ev)                       # code → trunc → cint → role
    by_cint = code_map.select("cint", "trunc", "role").unique().sort("cint")
    trunc_by_cint = by_cint["trunc"].to_list()
    role_by_cint = [int(x) for x in by_cint["role"].to_list()]
    # Carry is_stat on the *raw code* lookup rather than joining it per chunk: it rides
    # through the big join for one byte per event row, whereas a per-chunk join would
    # probe the far larger exploded event×prediction frame.
    raw_to_cint = code_map.select(
        "code", "cint", (pl.col("role") == ROLE_STAT).alias("is_stat"))

    ev_int = (
        ev.select(JOIN_ID, pl.col("time").cast(pl.Datetime("us")), "code",
                  pl.col("numeric_value").cast(pl.Float32))
        .join(blocks.lazy(), on=JOIN_ID, how="inner")       # drops events outside cohort
        .join(raw_to_cint.lazy(), on="code", how="inner")
        .select("jint", "chunk", "time", "cint", "numeric_value", "is_stat")
    )
    # Materialize the event side ONCE. Previously this stayed lazy and was re-filtered
    # inside the chunk loop, so the whole MEDS store was rescanned n_chunks times (plus
    # once more for time_since_first). The chunking below still bounds peak memory where
    # it matters — on the exploded join, not on this frame.
    ev_mat = ev_int.collect(engine="streaming")

    preds_int = (
        preds.select("r", JOIN_ID, "prediction_dttm")
        .join(blocks, on=JOIN_ID)
        .select(pl.col("r").cast(pl.UInt32), "jint", "chunk", "prediction_dttm")
    )

    # Aggregate one block-hash partition at a time → bounded peak memory.
    r_parts, c_parts, n_parts = [], [], []
    sr_parts, sc_parts, sv_parts = [], [], []
    stat_aggs = [getattr(pl.col("numeric_value"), s)().alias(s) for s in STATS]
    for k in tqdm(range(n_chunks), desc=desc or "featurize",
                  unit="chunk", leave=False):
        longk = (
            ev_mat.lazy().filter(pl.col("chunk") == k)
            .join(preds_int.lazy().filter(pl.col("chunk") == k), on="jint", how="inner")
            .filter(pl.col("time") < pl.col("prediction_dttm"))   # strict point-in-time
            .select("r", "cint", "numeric_value", "is_stat")
            .collect(engine="streaming")
        )
        # Split by role in one pass, then aggregate each side with only the statistics
        # it needs: a single combined group_by would compute the (sort-backed) median
        # for every medication code and discard it.
        parts = longk.partition_by("is_stat", as_dict=True)
        del longk

        cnt = parts.pop((False,), None)
        if cnt is not None and cnt.height:
            g = cnt.group_by("r", "cint").agg(pl.len().alias("n"))
            r_parts.append(g["r"].to_numpy())
            c_parts.append(g["cint"].to_numpy())
            n_parts.append(g["n"].to_numpy().astype("float32"))
            del g
        del cnt

        st = parts.pop((True,), None)
        if st is not None and st.height:
            # Nulls are ignored by every agg anyway; dropping them first shrinks the
            # frame the group_by has to hash.
            g = (st.filter(pl.col("numeric_value").is_not_null())
                 .group_by("r", "cint").agg(*stat_aggs))
            if g.height:
                sr_parts.append(g["r"].to_numpy())
                sc_parts.append(g["cint"].to_numpy())
                sv_parts.append(
                    np.column_stack([g[s].to_numpy() for s in STATS]).astype("float32"))
            del g
        del st, parts
        gc.collect()

    def _cat(parts, dtype, width=None):
        if parts:
            return np.concatenate(parts)
        return (np.empty(0, dtype) if width is None
                else np.empty((0, width), dtype))

    rr = _cat(r_parts, "uint32")
    cc = _cat(c_parts, "uint32")
    nn = _cat(n_parts, "float32")
    srr = _cat(sr_parts, "uint32")
    scc = _cat(sc_parts, "uint32")
    svals = _cat(sv_parts, "float32", width=len(STATS))
    del r_parts, c_parts, n_parts, sr_parts, sc_parts, sv_parts
    gc.collect()

    # Dense extra: hours from the encounter's first MEDS event to the prediction
    # time. Uses ALL of the encounter's events (not the point-in-time-filtered set)
    # — "first event" is the earliest record for that join_id. Per prediction row,
    # in prediction_ids order, so it aligns to the matrix rows built downstream.
    # Read off the materialized frame; ev_mat is pre-point-in-time and already
    # inner-joined to the cohort, so this is the same set the old full rescan saw.
    first_t = ev_mat.group_by("jint").agg(pl.col("time").min().alias("t0"))
    time_since_first = (
        preds_int.join(first_t, on="jint", how="left")
        .with_columns(
            ((pl.col("prediction_dttm") - pl.col("t0")).dt.total_seconds() / 3600.0)
            .alias("tse")
        )
        .sort("r")["tse"].fill_null(0.0).to_numpy().astype("float32")
    )
    del ev_mat, first_t
    gc.collect()

    return Counts(rr=rr, cc=cc, nn=nn, trunc_by_cint=trunc_by_cint,
                  prediction_ids=prediction_ids, split=split, n_rows=n_rows,
                  time_since_first=time_since_first,
                  srr=srr, scc=scc, svals=svals, role_by_cint=role_by_cint)


def counts_to_X(counts: Counts, vocab: list[str] | None = None,
                roles: list[int] | None = None
                ) -> tuple[np.ndarray, list[str], list[str]]:
    """Resolve vocabulary and build the dense feature matrix from cached ``Counts``.

    ``vocab=None`` fits the vocabulary on the train split (the default for
    training); otherwise the supplied vocab is used (scoring a new site against
    a saved model), with ``roles`` from the saved model when available so the
    column layout can never drift. Cheap relative to ``compute_counts``.

    Returns ``(X, prediction_ids, feature_names)`` where ``X`` is float32
    ``(n_rows, len(feature_names))``: **NaN** in ROLE_STAT columns with no
    measurement, genuine **0** in ROLE_COUNT/ROLE_ONEHOT columns with no events.
    """
    trunc_by_cint = counts.trunc_by_cint
    prediction_ids = counts.prediction_ids
    n_rows = counts.n_rows
    n_codes = len(trunc_by_cint)

    if vocab is None:
        # Full code space: every extracted (truncated) code gets its columns, even
        # with zero events in train (a constant column — harmless, trees never split
        # on it). Keeps the feature space complete and identical across tasks/sites.
        vocab = list(trunc_by_cint)
        roles = (list(counts.role_by_cint) if counts.role_by_cint is not None
                 else [code_role(c) for c in vocab])
    elif roles is None:
        roles = [code_role(c) for c in vocab]

    names = feature_names(vocab, roles)
    if not names:
        return np.zeros((n_rows, 0), dtype="float32"), prediction_ids, []

    # Column layout: each vocab entry claims len(STATS) columns if ROLE_STAT else 1.
    starts, p = [], 0
    for role in roles:
        starts.append(p)
        p += len(STATS) if role == ROLE_STAT else 1
    n_cols = p

    # cint → (first column, role); -1 for codes absent from the vocabulary.
    name_to_i = {code: i for i, code in enumerate(vocab)}
    start_of = np.full(n_codes, -1, dtype="int64")
    role_of = np.full(n_codes, -1, dtype="int8")
    for ci, trunc in enumerate(trunc_by_cint):
        j = name_to_i.get(trunc)
        if j is not None:
            start_of[ci] = starts[j]
            role_of[ci] = roles[j]

    # Missing means different things per role, so seed the two column families
    # differently and let the scatters below overwrite what was actually observed.
    X = np.zeros((n_rows, n_cols), dtype="float32")
    stat_cols = np.concatenate(
        [np.arange(starts[j], starts[j] + len(STATS))
         for j, role in enumerate(roles) if role == ROLE_STAT]
    ) if any(r == ROLE_STAT for r in roles) else np.empty(0, dtype="int64")
    if stat_cols.size:
        X[:, stat_cols] = np.nan

    # Count / one-hot block.
    if counts.cc.size:
        col = start_of[counts.cc]
        role = role_of[counts.cc]
        keep = (col >= 0) & (role != ROLE_STAT)
        if keep.any():
            vals = counts.nn[keep]
            # ROLE_ONEHOT is "ever occurred", so clamp here rather than in the cache.
            vals = np.where(role[keep] == ROLE_ONEHOT, np.minimum(vals, 1.0), vals)
            X[counts.rr[keep], col[keep]] = vals

    # Statistics block — each surviving row writes len(STATS) consecutive columns.
    if counts.scc is not None and counts.scc.size:
        col = start_of[counts.scc]
        keep = (col >= 0) & (role_of[counts.scc] == ROLE_STAT)
        if keep.any():
            rows = counts.srr[keep].astype("int64")[:, None]
            cols = col[keep][:, None] + np.arange(len(STATS))
            X[rows, cols] = counts.svals[keep]

    return X, prediction_ids, names


def _ids_hash(prediction_ids: list[str]) -> str:
    """Stable fingerprint of the cohort's prediction_ids (order-sensitive)."""
    h = hashlib.sha1()
    for pid in prediction_ids:
        h.update(str(pid).encode())
        h.update(b"\n")
    return h.hexdigest()


def _meta_path(path: str | Path) -> Path:
    return Path(path).with_name("features_meta.json")


def save_counts(path: str | Path, counts: Counts) -> None:
    """Persist ``Counts`` to ``features.npz`` + a ``features_meta.json`` sidecar.

    The npz holds the binary arrays (both COO blocks + prediction_ids + split); the
    JSON sidecar holds ``trunc_by_cint``/``role_by_cint``, the cache format, and a
    cohort fingerprint (id hash + n_rows) so ``load_counts`` can cheaply validate
    before reading the npz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tse = counts.time_since_first
    if tse is None:
        tse = np.zeros(counts.n_rows, dtype="float32")
    np.savez_compressed(
        path,
        rr=counts.rr, cc=counts.cc, nn=counts.nn,
        srr=counts.srr, scc=counts.scc, svals=counts.svals,
        prediction_ids=np.asarray(counts.prediction_ids, dtype=object).astype("U"),
        split=counts.split.astype("U"),
        time_since_first=tse.astype("float32"),
    )
    # np.savez appends .npz if missing — normalize so the meta sidecar sits beside it.
    saved = path if path.suffix == ".npz" else path.with_suffix(".npz")
    _meta_path(saved).write_text(json.dumps({
        "format": CACHE_FORMAT,
        "trunc_by_cint": counts.trunc_by_cint,
        "role_by_cint": counts.role_by_cint,
        "stats": list(STATS),
        "n_rows": counts.n_rows,
        "ids_hash": _ids_hash(counts.prediction_ids),
    }))


def load_counts(path: str | Path, task_df: pl.DataFrame) -> Counts | None:
    """Load cached ``Counts`` iff it matches ``task_df``'s cohort, else ``None``.

    Validation compares the cache format and the sidecar's ``n_rows`` +
    prediction-id hash against the current cohort, so a stale cache (different or
    resplit cohort, e.g. the train run's full cohort vs an infer holdout — or a
    pre-statistics count-only npz) is rejected and recomputed. The format check is
    not optional: a v1 npz has no ``svals`` and would otherwise load as a matrix
    whose every statistic column is silently missing.
    """
    path = Path(path)
    saved = path if path.suffix == ".npz" else path.with_suffix(".npz")
    meta_p = _meta_path(saved)
    if not (saved.exists() and meta_p.exists()):
        return None
    meta = json.loads(meta_p.read_text())
    if meta.get("format") != CACHE_FORMAT or meta.get("stats") != list(STATS):
        return None
    cohort_ids = task_df["prediction_id"].to_list()
    if meta.get("n_rows") != len(cohort_ids):
        return None
    if meta.get("ids_hash") != _ids_hash(cohort_ids):
        return None
    with np.load(saved, allow_pickle=False) as z:
        n_rows = int(meta["n_rows"])
        trunc_by_cint = list(meta["trunc_by_cint"])
        roles = meta.get("role_by_cint")
        return Counts(
            rr=z["rr"], cc=z["cc"], nn=z["nn"],
            srr=z["srr"], scc=z["scc"], svals=z["svals"],
            trunc_by_cint=trunc_by_cint,
            role_by_cint=(list(roles) if roles is not None
                          else [code_role(c) for c in trunc_by_cint]),
            prediction_ids=z["prediction_ids"].tolist(),
            split=z["split"].astype(object).astype(str),
            n_rows=n_rows,
            time_since_first=z["time_since_first"].astype("float32"),
        )


def count_features(events, task_df: pl.DataFrame, label_col: str,
                   vocab: list[str] | None = None, roles: list[int] | None = None,
                   n_chunks: int = 8, desc: str | None = None
                   ) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (X, prediction_ids, feature_names). Row order = prediction_ids.

    Thin wrapper: ``compute_counts`` (heavy join) then ``counts_to_X`` (vocab +
    dense matrix). Kept for backward compatibility; the CLI splits the two so the
    join output can be cached. The vocabulary is fit on the train split, or supplied
    via ``vocab=``. Extraction is the single source of truth: whatever the ELF
    config extracted into MEDS is exactly what can be featured — to drop a
    domain, remove it from flair_elf_config.yaml.
    """
    counts = compute_counts(events, task_df, n_chunks=n_chunks, desc=desc)
    return counts_to_X(counts, vocab=vocab, roles=roles)
