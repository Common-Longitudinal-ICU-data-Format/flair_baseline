# Transfer learning and local training at external sites

**Date:** 2026-07-26
**Status:** Approved
**Branch:** `transfer-learning` (cut from `main`)

## Problem

Today the baseline supports exactly two ways to get a number out of a site:

- `train` — fit XGBoost on the local train split, report on the local test split.
  This is what MIMIC runs; its model is what ships.
- `infer` — score a shipped model on another site's 25% holdout.

That gives an external site one answer: *how well does the MIMIC model transfer,
untouched?* It cannot answer the two questions that follow immediately:

1. **Transfer learning** — does continuing to train the MIMIC model on local data
   beat the frozen MIMIC model?
2. **Local training** — does a model trained only on local data beat both?

`train` already does (2) mechanically, but it writes to the same
`report/` directory that `infer` does, so a site cannot hold both results at once.
Nothing distinguishes the three answers on disk.

## Scope

All edits land in `flair_baseline/`. `flair_benchmark/` receives **zero changes** —
it is called through `build_report()` exactly as it is today.

Featurization, cohort building, the ELF vocabulary, and the 75/25 block split are
untouched. This spec adds one training mode and a per-model-kind output routing.

Out of scope: k-fold cross-validation, meta-analysis or pooling across sites, and
any change to the report schema.

## Design

### 1. One new axis: `kind`

A single concept carries the whole feature: every artifact that is *per model*
rather than *per site* gains a `kind` segment.

```
kind ∈ {external_validation, transfer, local}
```

It routes exactly three paths:

```
<site>_baseline_phi/<task>/preds_<kind>.parquet
<site>_baseline_non_phi_for_upload/<task>/report/<kind>/
<site>_baseline_models/<task>/<kind>/{model,vocab,params}.json
```

Everything else stays where it is. `cohort.parquet`, `features.npz`,
`codes.parquet` and `table1.json` describe **the site's data, not a model**, so
they are not duplicated per kind — that is the whole reason the split lives under
`report/` rather than at the top of the non-PHI root.

Per-kind `preds_<kind>.parquet` is not cosmetic: without it the second pipeline to
run at a site silently overwrites the first one's predictions, and the report built
from it would be attributed to the wrong model.

### 2. Commands

Three commands, one per kind. Only one of them is new code.

| command | alias | fits a model? | base model | rows scored | report dir |
|---|---|---|---|---|---|
| `local-training` | `train` | yes — HPO | — | all | `report/local/` |
| `transfer-learning` | — | yes — HPO | `--models-dir` | all | `report/transfer/` |
| `external-validation` | `infer` | no | `--models-dir` | `split == "test"` | `report/external_validation/` |

Both fitting commands expose the same `--hpo/--no-hpo` and `--hpo-trials`
(default 30) flags that `train` has today, with the same meaning.

`local-training` and `external-validation` are **renames** of the existing `train`
and `infer`. The old names stay as hidden Typer aliases so `run_mimic_train.sh`
and any site's existing scripts keep working. The only genuinely new training path
is `transfer-learning`.

Site roles follow from this without any enforcement in code:

- **MIMIC (source site):** runs `local-training` only. Ships
  `mimic_baseline_models/<task>/local/`.
- **External site:** runs all three. All three report on the *same* local 25%
  holdout, so the three JSON bundles are directly comparable.

### 3. Transfer learning mechanism

Continued boosting: keep the MIMIC ensemble, append trees fit on local train rows.

**Column space.** The base trees index into the source site's column layout, so
transfer loads the **shipped** `vocab.json` from `--models-dir`, not the local
committed `vocab.json` — the same thing `infer` does today. In practice the two
are identical (the vocabulary is a fixed committed file shared by all sites), but
resolving against the shipped copy is what makes that an invariant rather than a
coincidence. If the resulting matrix width disagrees with the base booster's
feature count, fail loudly rather than predicting garbage.

**HPO.** `xgb.cv()` accepts no `xgb_model=` argument, so continued boosting cannot
be cross-validated directly. It can be expressed equivalently: boosting on top of
an existing ensemble is boosting with that ensemble's raw output as a
per-row offset. So the search runs as

```
m = base.predict(X[train], output_margin=True)
dtrain.set_base_margin(m); base_score = 0
```

with the **existing `tune_xgb` search space and trial count, unchanged**. No new
search space to maintain, and the tuning objective is the real one (5-fold CV
AUROC on local train rows only, never touching the test split).

**Final fit.** The chosen params are fit with `xgb_model=base_booster`, which
produces one self-contained `model.json` holding MIMIC trees plus local trees.
Scoring it is an ordinary predict — no companion model has to travel with it, and
`external-validation` at a third site could consume it unmodified.

`scale_pos_weight` stays at 1.0, matching `local-training`, for the calibration
reason already documented in `train.py`.

### 4. Guards

- `transfer-learning` and `local-training` fit on the local **train** split, so they
  require full-cohort features. `load_counts()` already rejects a holdout-only
  `features.npz` through its prediction-id hash; the generic
  "no matching features.npz" message becomes a specific
  "re-run `featurize` without `--holdout-only`".
- `--holdout-only` remains fully valid — it is the cheap path for a site that only
  wants `external-validation`.
- **`external-validation` must accept either featurization scope.** It previously
  filtered the cohort to the test split *before* loading features, which the
  prediction-id hash then rejects at any site that featurized the full cohort —
  i.e. at every site running all three kinds, defeating the entire point of a
  shared holdout. It now tries the full cohort first and falls back to the test
  split, so one `featurize` pass feeds all three commands. The headline metric is
  the test split either way; when the full cohort is scored, the extra rows land
  in the report's train bin, which for a frozen model is simply more external data.
- `transfer-learning` without `--models-dir`, or with no model under
  `<models-dir>/<task>/`, exits with a clear message rather than falling back to
  training from scratch. Silently degrading a transfer run into a local run would
  produce a mislabelled report.

### 5. Back-compatibility

Model bundles shipped before this change are flat: `<task>/model.json`. The
resolver tries `<task>/local/model.json` first and falls back to `<task>/model.json`,
so existing bundles keep scoring.

**Known cost:** MIMIC's current non-PHI bundle has its report at `report/*.json`;
the new layout puts it at `report/local/*.json`. The existing MIMIC report needs to
be regenerated into the new location. This is a report rebuild from the retained
PHI predictions, not a retrain.

## Components and boundaries

| unit | responsibility | depends on |
|---|---|---|
| `layout.TaskPaths` | sole owner of the `kind` → path mapping (`report_dir(kind)`, `model_dir(kind)`, `preds(kind)`) | nothing |
| `train.tune_xgb` | param search; gains an optional `base_margin` | xgboost, optuna |
| `train.train_and_score` | fit-or-load + score; gains an optional `base_model` | above |
| `cli` | argument parsing, guards, stage sequencing, report invocation | above |

No new module. `train.py` stays the only place that knows what XGBoost is;
`layout.py` stays the only place that knows where files go. A reader should be
able to answer "where does a transfer report land?" by reading `layout.py` alone.

## Error handling

Every new failure is a `typer.Exit(1)` with a message naming the command to run
next — matching the existing style (`_read_cohort`, `_load_vocab`,
the all-zero-features guard). Specifically: missing base model, base/local feature
width mismatch, and holdout-only features supplied to a fitting command.

The existing viz-failure tolerance in `_report` is preserved as-is for all three
kinds: report JSONs written but PNG rendering failed is a warning, not a failure.

## Testing

- `TaskPaths` returns the three expected per-kind paths and leaves `codes`,
  `table1`, `cohort`, `features` un-suffixed.
- Old-flat vs new-nested model bundle both resolve.
- `tune_xgb` with a `base_margin` runs and returns a param dict.
- Transfer on a small synthetic task: resulting booster has strictly more trees
  than the base, and its predictions differ from the base model's.
- A holdout-only `features.npz` passed to `local-training` / `transfer-learning`
  exits non-zero with the `--holdout-only` message.
- End-to-end on one task at one site: all three kinds produce populated,
  non-overlapping report directories and three distinct preds files.

## Documentation

`flow.md` already documents a monolithic `train`/`infer` flow that no longer
matches the staged CLI. It is rewritten alongside this change to describe the four
staged commands plus the three model kinds, including the external-site run order.
