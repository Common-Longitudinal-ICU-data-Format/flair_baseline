# Migrate flair-baseline onto the reorganized flair-benchmark

**Date:** 2026-07-26
**Status:** Approved
**Branch:** `baseline-import`

## Problem

This is a **forward migration, not a repair.** `baseline-import` is currently
internally consistent and working: its bundled wheel
(`wheels/flair_benchmark-0.0.1-py3-none-any.whl`, 168 KB) is *pre-reorg* —
it ships `_clif.py`, `_stitch.py`, `_table1.py`, `features/fe_meds.py`,
`report/discrimination.py`, `report/peak.py`, `report/leadtime.py` — while
already being *post-task-renaming* (4 de-numbered tasks). `run_mimic_train.sh`
was a genuine run against it.

Benchmark commit `6056736` ("refactor: reorganize flair_benchmark into
pipeline-stage packages") restructured the package. The breakage is therefore
**created by rebuilding the wheel from current benchmark HEAD**, which is the
goal here. Two things break at that moment:

1. **Import paths.** `_clif`, `_stitch`, `_table1`, `_constants`, and
   `features.fe_meds` no longer exist. 14 import sites raise
   `ModuleNotFoundError`.
2. **The report bundle was consolidated.** `discrimination.json` — which the
   baseline reads for its headline AUROC — no longer exists.

Already handled on this branch, and **out of scope**:

- `config.py` gained `available_tasks()` and an `EXCLUDED_TASKS` choke point; the
  dead `taskN` prefix branch is gone and the removed sepsis task is accounted for.
- `layout.report_mode()` already reads `get_task(name).META["report_mode"]`
  rather than the deleted `TASK_POLICY`. Both the old wheel and current HEAD use
  `REPORT_MODES = {"episodic", "continuous"}` with identical per-task values, so
  this function needs **no change** and keeps working across the migration.
- The baseline version is already `0.1.0`.

## Scope

`flair_benchmark/` receives **zero changes**. It is CLI + API driven; the
baseline is a legitimate API consumer. All edits land in `flair_baseline/` on the
`baseline-import` branch.

The benchmark version stays at `0.0.1`. Wheel freshness is maintained by an
explicit force-reinstall ritual rather than a version bump. The FLAIR working
tree is clean at `add_new_task`, so the build is reproducible.

## Design

### 1. Import repointing

| Old (deleted) | New | Sites |
|---|---|---|
| `flair_benchmark._clif` | `flair_benchmark.cohort.clif` | 8 |
| `flair_benchmark._stitch` | `flair_benchmark.cohort.stitch` | 3 |
| `flair_benchmark._table1` | `flair_benchmark.cohort.table1` | 1 |
| `flair_benchmark.features.fe_meds` | `flair_benchmark.features.extractors` | 2 |

14 total: 12 in `src/flair_baseline/cli.py`, 2 in `tests/test_shared_meds.py`.

`flair_benchmark.tasks`, `flair_benchmark.meds_etl.build_codes_table`, and
`flair_benchmark.report.build_report` are unchanged — `config.py` and
`layout.py` need no import edits.

Every function the baseline calls survived with a compatible signature:
`build_shared_meds`, `load_elf_config`, `members_of_joins`,
`load_or_build_encounter_index`, `build_codes_table`, `generate_table1`,
`read_clif_config`, `build_report`. `tests/test_shared_meds.py` imports the
private `_batches`, which survived at `features.extractors._batches`.

### 2. Report bundle consolidation

`REPORT_SCHEMA` went 3 → 4. The bundle collapsed from 5–6 files to 2:

| Mode | Old (schema 3) | New (schema 4) |
|---|---|---|
| episodic | `discrimination`, `calibration`, `dca`, `fairness`, `operating_points` | `overall`, `hospitalization_level` |
| continuous | `discrimination`, `calibration`, `fairness`, `leadtime`, `operating_points`, `kpi` | `landmark`, `hospitalization_level` |

**No metric was dropped** — the content moved inside the two files:

- `overall.json` → `discrimination` (AUROC/AUPRC + CIs) and `calibration`
- `landmark.json` → `pooled` (pair-weighted cross-landmark aggregate) and
  `by_landmark`
- `hospitalization_level.json` → the 99-threshold sweep, Youden, `fairness`, and
  `net_benefit` (the former DCA), for both modes

Verified payload shapes:

```
overall.json   { task_type, metadata, n, prevalence,
                 discrimination: { auroc, auroc_ci, auprc, ... },
                 calibration: {...} }

landmark.json  { task_type, metadata, prevalence,
                 pooled: { auroc, auprc, brier, calibration_slope, ... },
                 by_landmark: [ { hospitalization_time, n, auroc, ... } ] }
```

#### 2a. `cli._report_auroc()`

The old reader looked for the file `discrimination.json` and the keys `metrics`
and `landmarks[0].metrics`. All three names are gone. Rewrite to take `mode` as
an argument (the caller already knows it) and read:

- `episodic` → `overall.json` → `discrimination.auroc`
- `continuous` → `landmark.json` → `pooled.auroc`

For continuous tasks this **deliberately changes the headline number**: the old
code printed the lead-0 landmark, whereas `pooled` is the benchmark's own
pair-weighted cross-landmark aggregate and is the more faithful single figure.
Drop the `" lead-0"` suffix from the log line accordingly. MIMIC numbers will not
be comparable to previous runs of this baseline.

Both branches must tolerate a missing or `None` AUROC — the benchmark emits that
legitimately for degenerate splits (single label class, or a landmark below
`min_cell_count`).

#### 2b. Documentation

The README's upload-check block lists `report/*.json` generically, but the task
table and any enumerated filenames must reflect the 2-file bundle. Note the
`report_schema: 4` stamp so a site can tell an old upload from a new one.

### 3. Packaging and venv

`flair_baseline/pyproject.toml`:

- `flair-benchmark` → `flair-benchmark[viz]`. The baseline passes `viz=True` into
  `build_report`, which needs matplotlib; `build_report` catches the
  `ImportError`, so `--viz` currently no-ops **silently**.
- Add `scipy>=1.10`. `featurize.py` and `train.py` import `scipy.sparse` but it
  is only present as a scikit-learn transitive.

Wheel rebuild and sync, to be documented verbatim in the README:

```bash
# from the FLAIR repo root
rm -f flair_baseline/wheels/*.whl
uv build --wheel --out-dir flair_baseline/wheels

# from flair_baseline/
uv sync --reinstall-package flair-benchmark
```

The baseline gets its own `flair_baseline/.venv`, independent of the FLAIR root
venv. `.venv/` is already gitignored; `wheels/*.whl` is force-included.

### 4. `run_mimic_train.sh`

The script hardcodes `cd /Users/sudo_sage/Documents/WORK/FLAIR_PROJECT/flair_baseline`,
a path that does not exist on this machine. Replace the hardcoded `cd` with a
`cd "$(dirname "$0")"` so the script is location-independent, and add a
`--no-hpo` pass-through. Its resumable done-stamp design
(`.mimic_run_stamps/<task>.done`) is worth keeping as-is.

### 5. MIMIC validation

Create `flair_baseline/config/clif_config_mimic.json` — the name
`run_mimic_train.sh` already expects, gitignored via `clif_config*.json`:

```json
{
    "site": "mimic",
    "data_directory": "/Users/sudo_sage/Downloads/work/clif_m",
    "filetype": "parquet",
    "timezone": "US/Eastern",
    "stitch_time_interval_hours": 6,
    "cache_directory": "./output"
}
```

Start from a **clean output root** so no Int64-era artifact is inherited (see the
dtype-flip analysis below) — remove `mimic_baseline_phi/`,
`mimic_baseline_non_phi_for_upload/`, `mimic_baseline_models/`,
`.mimic_run_stamps/`, and the clifpy `output/` cache. Then run all 4 tasks with
`--reuse` off (the default) and no HPO:

```bash
uv run flair-baseline prepare --clif-config config/clif_config_mimic.json --out . --pmc
uv run flair-baseline train   --clif-config config/clif_config_mimic.json --out . --no-hpo --viz
```

`build-vocab` is deliberately **not** run. The committed `vocab.json` is a fixed
cross-site vocabulary by design; regenerating it from MIMIC would defeat the
purpose of interchangeable model bundles.

Success criteria:

- 4 cohorts written, each with train and test splits and a `table1.json`.
- One shared MEDS store under `mimic_baseline_phi/_shared/MEDS/` with
  `part-*.parquet` and a `manifest.json`.
- Per-task `features.npz` and `codes.parquet`.
- Per-task report dir containing `overall.json` or `landmark.json`, plus
  `hospitalization_level.json`, stamped `report_schema: 4`, with a non-null test
  AUROC.
- Viz PNGs under each report dir's `viz/`.
- `mimic_baseline_models/<task>/` containing `model.json`, `vocab.json`,
  `params.json`.

### 6. Tests

- Repoint the two `flair_benchmark` imports in `tests/test_shared_meds.py`.
- Add a regression test asserting `layout.report_mode(t)` returns a member of
  `flair_benchmark.constants.REPORT_MODES` for every task in `available_tasks()`.
  This turns a future benchmark reorg into a fast `pytest` failure instead of a
  crash hours into a MIMIC run.
- Add a test that `_report_auroc` returns `(None, ...)` rather than raising when
  the report dir is empty or the AUROC is `None`.
- Run the full suite in the baseline venv.

## Risk outcomes

1. **Episodic one-row-per-stay guard** — *cleared.* `build_report` raises when an
   episodic task has more prediction rows than stays, and
   `extubation_failure_24h` is per-extubation-episode. Measured on the MIMIC test
   split: `extubation_failure_24h` 5,254 rows / 5,254 stays and
   `icu_readmission` 13,256 / 13,256 — exactly 1:1. `tasks/_finalize.py` already
   dedups, so the guard does not trip and no `--mode continuous` fallback is
   needed.
2. **`[viz]` extra through a local-path wheel source** — *cleared.* The rebuilt
   wheel's METADATA carries `Provides-Extra: viz` and
   `Requires-Dist: matplotlib>=3.7.0; extra == 'viz'`; matplotlib 3.11.0 resolved
   into the baseline venv.
3. **Cohort schema drift** — still open until the featurize and report stages
   complete. Would surface as a `compute_counts` failure or a `build_report`
   preds-schema rejection (`hospitalization_join_id`, `prediction_id`, `split`,
   label column, `y_prob`).

### Resolved during design: the MEDS dtype flip

`MEDS_SCHEMA` changed `subject_id` and `hospitalization_id` from `pl.Int64` to
`pl.Utf8` in the reorg, with the rationale inline: *"Utf8, never Int64: a numeric
cast turns every alphanumeric site id into null."* This is a correctness fix for
sites with non-numeric IDs, but a **silent-failure hazard**, because a Polars
join across mismatched dtypes returns zero matched rows rather than raising. An
all-zero count matrix would still train, still report, and still emit a
plausible AUROC near 0.5.

Analysis says the exposure is narrow, not absent:

- `featurize.py` joins only on `hospitalization_join_id` and `code` — never on
  `subject_id` or `hospitalization_id`.
- New `cohort/stitch.py:attach_join_id` casts `hospitalization_id` to `Utf8` on
  both sides and documents `hospitalization_join_id` as `Utf8`, so new-wheel
  events and new-wheel cohorts agree.

The real risk is therefore **mixing eras**: an Int64-era `_shared/MEDS/` picked up
by `--reuse`, or a `cohort.parquet` written by the old wheel and read by new
code. Mitigations, both required:

- Run the MIMIC validation into a **clean output root**, with `--reuse` off, so
  no pre-migration artifact is inherited. Delete any existing
  `mimic_baseline_phi/`, `mimic_baseline_non_phi_for_upload/`,
  `mimic_baseline_models/`, `.mimic_run_stamps/`, and the clifpy `output/` cache.
- Add an assertion in `_do_featurize` that the post-join count matrix is
  non-empty, so a dtype or key mismatch fails loudly instead of training on
  zeros. This is the guard that makes the whole class of error visible, not just
  this instance.

## Out of scope

- Adding CLI commands to the benchmark for the stitch index, shared MEDS ETL, or
  codes table. The benchmark is CLI + API; there is no CLI-only requirement.
- Bumping the benchmark version or publishing to PyPI.
- Restoring `sepsis_abx_6h`, or the task-inventory and `report_mode` work already
  completed on `baseline-import`.
- Re-tuning hyperparameters or changing the feature design.
