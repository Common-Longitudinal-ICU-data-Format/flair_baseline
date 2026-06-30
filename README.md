# flair-baseline

XGBoost count-feature baseline for the FLAIR ICU benchmark.

The model is trained once by the model owner, then each site runs inference on its own CLIF data and uploads only the non-PHI report folder.

Patient-level data never leaves the site.

## Quick Start For CLIF Sites

Use this section if you are a participating CLIF site and only need to run inference.

### 1. Install Once

Run this after cloning the repo, or whenever dependencies change:

``` bash
uv sync
```

### 2. Create Your Local Config

``` bash
cp config/clif_config.template.json config/clif_config.json
```

Edit `config/clif_config.json` with your local site name and CLIF data location.

Use generic values like this:

``` json
{
    "site": "example_site",
    "data_directory": "/path/to/local/clif/data",
    "filetype": "parquet",
    "timezone": "US/Central",
    "stitch_time_interval_hours": 6,
    "cache_directory": "./output"
}
```

`site` becomes the prefix for output folders, for example `example_site_baseline_non_phi_for_upload/`.

`data_directory` should point to the folder containing your local CLIF tables, for example files named `clif_patient.parquet`, `clif_hospitalization.parquet`, `clif_vitals.parquet`, and related CLIF tables.

`config/clif_config.json` is ignored by git because it is site-specific.

### 3. Add The Model Bundle

Download or copy the trained model bundle into the repository root.

The expected folder layout is:

``` text
mimic_baseline_models/
  task1_icu_daily_mortality/model.json
  task1_icu_daily_mortality/vocab.json
  task2_icu_daily_ltach/model.json
  task2_icu_daily_ltach/vocab.json
  task3_extubation_failure_24h/model.json
  task3_extubation_failure_24h/vocab.json
  task4_sepsis_abx_6h/model.json
  task4_sepsis_abx_6h/vocab.json
  task5_icu_readmission/model.json
  task5_icu_readmission/vocab.json
```

### 4. Run Inference

Recommended site command:

``` bash
uv run flair-baseline prepare --clif-config config/clif_config.json --out . --holdout-only --reuse --pmc
uv run flair-baseline infer --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
```

What this does:

`prepare` runs `build-cohorts`, `build-data`, and `featurize` in one command.

`--holdout-only` extracts and featurizes only the deterministic 25% test split used for external inference.

`--reuse` reuses the shared MEDS output if the manifest still matches.

`--pmc` stands for `poor-man's-compute`. It uses batched ETL to reduce peak memory usage. It is slower but safer on lower-memory machines.

When `--pmc` is used, the default batch size is `4000` encounters. You can tune it with `--batch-size N`. Smaller values use less memory but run slower; larger values may run faster but need more memory.

`infer` loads the shipped models, scores the holdout rows, and writes report JSONs and optional PNG visualizations.

### 5. Upload Only The Non-PHI Folder

After inference, upload only:

``` text
example_site_baseline_non_phi_for_upload/
```

Do not upload:

``` text
example_site_baseline_phi/
```

The PHI folder contains cohorts, features, shared MEDS events, and predictions. It should stay local.

## Output Folders

Every run writes site-prefixed folders under `--out`.

| Folder | Contains | Upload? |
|------------------------|------------------------|------------------------|
| `<site>_baseline_phi/` | `cohort.parquet`, shared `_shared/MEDS/`, `features.npz`, `preds.parquet` | No |
| `<site>_baseline_non_phi_for_upload/` | `codes.parquet`, `table1.json`, report JSONs, optional report PNGs | Yes |
| `mimic_baseline_models/` | Public trained model bundle used for inference | No |

Quick upload check:

``` text
<site>_baseline_non_phi_for_upload/<task>/codes.parquet
<site>_baseline_non_phi_for_upload/<task>/table1.json
<site>_baseline_non_phi_for_upload/<task>/report/*.json
<site>_baseline_non_phi_for_upload/<task>/report/viz/*.png
```

The upload folder should not contain `cohort.parquet`, `_shared/MEDS/`, `features.npz`, or `preds.parquet`.

## Common Site Commands

### Run All Tasks, Low-Memory Safe

This is the recommended default for most sites:

``` bash
uv run flair-baseline prepare --clif-config config/clif_config.json --out . --holdout-only --reuse --pmc
uv run flair-baseline infer --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
```

To lower memory further, reduce the batch size:

``` bash
uv run flair-baseline prepare --clif-config config/clif_config.json --out . --holdout-only --reuse --pmc --batch-size 1000
uv run flair-baseline infer --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
```

### Run All Tasks Without Batched ETL

Use this if the machine has enough memory and you want a simpler single-pass ETL:

``` bash
uv run flair-baseline prepare --clif-config config/clif_config.json --out . --holdout-only --reuse
uv run flair-baseline infer --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
```

### Run One Task Only

Add `--task task1`, `--task task2`, `--task task3`, `--task task4`, or `--task task5` to both commands.

``` bash
uv run flair-baseline prepare --clif-config config/clif_config.json --out . --holdout-only --reuse --pmc --task task1
uv run flair-baseline infer --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz --task task1
```

### Run The Stages Manually

Use this only if you want more control or need to restart from a specific stage:

``` bash
uv run flair-baseline build-cohorts --clif-config config/clif_config.json --out .
uv run flair-baseline build-data --clif-config config/clif_config.json --out . --holdout-only --reuse --pmc
uv run flair-baseline featurize --clif-config config/clif_config.json --out . --holdout-only
uv run flair-baseline infer --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
```

## What The Pipeline Does

The inference workflow has four stages.

| Stage | Command | Output |
|------------------------|------------------------|------------------------|
| 1 | `build-cohorts` | Per-task cohorts with train/test split and `table1.json` |
| 2 | `build-data` | One shared CLIF-to-MEDS store under `<site>_baseline_phi/_shared/MEDS/` |
| 3 | `featurize` | Sparse count features per task plus uploadable `codes.parquet` |
| 4 | `infer` | Predictions, report JSONs, and optional visualizations |

The `prepare` command runs stages 1 through 3.

At external sites, `--holdout-only` keeps the expensive ETL and featurization scoped to the 25% test split. The full cohort is still built so `table1.json` can describe the full local cohort.

## Tasks

| Task    | Name                               | Report mode |
|---------|------------------------------------|-------------|
| `task1` | ICU daily in-hospital mortality    | Landmark    |
| `task2` | ICU daily LTACH discharge          | Landmark    |
| `task3` | Extubation failure within 24 hours | Episodic    |
| `task4` | Sepsis antibiotics within 6 hours  | Peak        |
| `task5` | ICU readmission                    | Episodic    |

Report modes:

`landmark` reports metrics at lead-time landmarks and includes `leadtime.json`.

`episodic` reports one prediction per stay.

`peak` evaluates each stay by peak risk and omits calibration and DCA because peak-risk calibration is not meaningful.

## Feature Inputs

The feature domains are controlled by `flair_elf_config.yaml`.

Included domains:

| Domain    | Description                                 |
|-----------|---------------------------------------------|
| `VITAL`   | Vitals                                      |
| `LAB`     | Labs                                        |
| `RESP`    | Respiratory support                         |
| `PA`      | Patient assessments, including GCS and RASS |
| `MED_CON` | Continuous medications                      |
| `MED_INT` | Intermittent medications                    |

The ELF config is the single source of truth for what is extracted into MEDS and what can be used as count features.

Discharge diagnoses are intentionally not used as features because they are assigned after the stay and can leak outcome information.

## Model Owner Workflow

Use this section only if you are training or refreshing the source model bundle.

Set `config/clif_config.json` to the source training site and run:

``` bash
uv run flair-baseline build-cohorts --clif-config config/clif_config.json --out .
uv run flair-baseline build-data --clif-config config/clif_config.json --out .
uv run flair-baseline build-vocab --clif-config config/clif_config.json --out .
uv run flair-baseline featurize --clif-config config/clif_config.json --out .
uv run flair-baseline train --clif-config config/clif_config.json --out . --viz
```

This writes:

``` text
<training_site>_baseline_phi/
<training_site>_baseline_non_phi_for_upload/
<training_site>_baseline_models/
```

Publish only the model bundle folder for sites to use for inference.

`build-vocab` is for maintainers/model owners. CLIF inference sites should use the vocabulary shipped in the model bundle and should not regenerate it.

## Configuration Reference

`config/clif_config.json` fields:

| Field | Meaning |
|------------------------------------|------------------------------------|
| `site` | Short site label used to prefix output folders |
| `data_directory` | Local path to CLIF tables |
| `filetype` | Usually `parquet` |
| `timezone` | Local timezone used for CLIF datetimes |
| `stitch_time_interval_hours` | Gap used for encounter stitching, usually `6` |
| `cache_directory` | Local cache directory used by CLIF tooling |

The committed `vocab.json` and the per-task `vocab.json` files in the model bundle keep feature columns aligned across sites.

## Troubleshooting

### Do I Need `uv sync` Every Time?

No. Run `uv sync` once after cloning or when dependencies change. If `uv run flair-baseline ...` works, the environment is already ready.

### The ETL Prints Datetime Warnings

Warnings like this may appear:

``` text
Naive datetime localized to US/Central. Please verify this is correct.
```

These are informational unless the command exits with an error. Confirm that `timezone` in `config/clif_config.json` matches the local CLIF data conventions.

### The ETL Prints Medication Unit Warnings

Warnings about missing medication categories or unsupported preferred-unit conversion can appear when a site does not contain certain medications or units. These are commonly non-fatal. If the command completes, the pipeline continued with the available mapped data.

### Restart After A Failed Run

If cohorts were built successfully, you can rerun the recommended commands. `--reuse` will reuse matching shared MEDS artifacts when possible.

``` bash
uv run flair-baseline prepare --clif-config config/clif_config.json --out . --holdout-only --reuse --pmc
uv run flair-baseline infer --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
```

### Need To Save Memory?

Keep `--pmc` (`poor-man's-compute`) on the `prepare` command. It batches the shared MEDS extraction into part files so peak RAM is bounded by one batch instead of the whole cohort.

The default `--pmc` batch size is `4000` encounters. If memory is still tight, set a smaller batch size:

``` bash
uv run flair-baseline prepare --clif-config config/clif_config.json --out . --holdout-only --reuse --pmc --batch-size 1000
```

Use a larger `--batch-size` only if the machine has enough memory.

You can also run one task at a time with `--task taskN`.

## Development Notes

The FLAIR benchmark library is bundled as a wheel under `wheels/` and pinned in `pyproject.toml`.

Maintainers updating the bundled library should rebuild the wheel, refresh the lockfile if needed, and reinstall the package:

``` bash
uv build --wheel --out-dir flair_baseline/wheels
rm -f flair_baseline/wheels/.gitignore
rm uv.lock
uv lock
uv sync --reinstall-package flair-benchmark
```

If the package version or wheel filename changes, update `[tool.uv.sources]` in `pyproject.toml`.