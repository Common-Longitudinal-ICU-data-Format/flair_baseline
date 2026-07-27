# flair-baseline

XGBoost point-in-time feature baseline for the FLAIR ICU benchmark.

Features summarize every ELF event that happened **strictly before** each prediction time, aggregated per code by value type: numeric codes contribute `min`/`max`/`mean`/`median`, medications contribute exposure counts, and categorical respiratory settings contribute presence indicators. See [Feature Engineering](#feature-engineering) for the full contract.

The model is trained once by the model owner, then each site evaluates it on its own CLIF data and uploads only the non-PHI report folder.

Patient-level data never leaves the site.

A site can produce up to three answers, all scored on the same local 25% holdout so they are directly comparable:

| kind | command | question it answers |
|------------------------|------------------------|------------------------|
| `external_validation` | `external-validation` | How well does the shipped model transfer to us, untouched? |
| `transfer` | `transfer-learning` | Does continuing to train it on our data help? |
| `local` | `local-training` | Does a model trained only on our data do better? |

## Quick Start For CLIF Sites

Use this section if you are a participating CLIF site.

**Decide first: validation only, or all three?** It changes one flag in the data preparation step, and preparing the wrong way means redoing the ETL.

|                        | validation only       | all three                |
|------------------------|-----------------------|--------------------------|
| prepare flag           | `--holdout-only`      | *(omit it)*              |
| ETL + featurize covers | 25% test split        | full cohort              |
| roughly                | 1x                    | 4x the data, 4x the time |
| you can then run       | `external-validation` | all three commands       |

`external-validation` works with either preparation. The two fitting commands need the train split, so they require the full-cohort preparation. If you prepare holdout-only and later want all three, the ETL has to run again — `--reuse` will correctly refuse to reuse a holdout store for a full-cohort run.

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
  icu_daily_mortality/local/{model.json,vocab.json,params.json}
  icu_daily_ltach/local/{model.json,vocab.json,params.json}
  extubation_failure_24h/local/{model.json,vocab.json,params.json}
  icu_readmission/local/{model.json,vocab.json,params.json}
```

Older bundles are flat (`<task>/model.json`, `<task>/vocab.json`) and still work — the commands look in `<task>/local/` first, then fall back.

### 4. Run External Validation

Recommended site command:

**Validation only:**

``` bash
uv run flair-baseline prepare --clif-config config/clif_config.json --out . --holdout-only --reuse --pmc
uv run flair-baseline external-validation --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
```

**All three** — note that `--holdout-only` is absent, and `prepare` runs only once:

``` bash
uv run flair-baseline prepare --clif-config config/clif_config.json --out . --reuse --pmc

uv run flair-baseline external-validation --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
uv run flair-baseline transfer-learning   --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
uv run flair-baseline local-training      --clif-config config/clif_config.json --out . --viz
```

What this does:

`prepare` runs `build-cohorts`, `build-data`, and `featurize` in one command. All three model commands read its output; none of them re-run the ETL.

`--holdout-only` extracts and featurizes only the deterministic 25% test split. Omit it if you want the two fitting commands.

`--reuse` reuses the shared MEDS output if the manifest still matches. The manifest records the scope, so a holdout-only store is correctly rejected for a full-cohort run rather than silently reused.

`--pmc` stands for `poor-man's-compute`. It uses batched ETL to reduce peak memory usage. It is slower but safer on lower-memory machines.

When `--pmc` is used, the default batch size is `4000` encounters. You can tune it with `--batch-size N`. Smaller values use less memory but run slower; larger values may run faster but need more memory.

`external-validation` loads the shipped models, scores them frozen, and writes reports into `report/external_validation/`. It accepts either preparation scope.

`transfer-learning` keeps the shipped model's trees and appends new ones fit on your train split, into `report/transfer/` and `<site>_baseline_models/<task>/transfer/`.

`local-training` fits a fresh model on your train split alone, into `report/local/` and `<site>_baseline_models/<task>/local/`.

Both fitting commands tune hyperparameters by default: 30 Optuna trials scored by 5-fold CV on your train rows only. Use `--no-hpo` to skip the search (much faster, uses hand-tuned defaults) or `--hpo-trials N` to change the budget. Budget roughly 10–60 minutes per task with HPO on, depending on cohort size.

`train` and `infer` still work as hidden aliases for `local-training` and `external-validation`, so existing site scripts do not need editing.

### 4c. What You Need Before Running

| requirement | detail |
|------------------------------------|------------------------------------|
| CLIF tables | the domains in `flair_elf_config.yaml`: patient, hospitalization, adt, vitals, labs, respiratory_support, patient_assessments, medication_admin_continuous/intermittent |
| model bundle | `mimic_baseline_models/` copied into the repo root (§3) |
| config | `config/clif_config.json` with your `site` and `data_directory` (§2) |
| disk | the shared MEDS store dominates; roughly 0.5–1 GB per 75k encounters, more for a full-cohort run |
| time | validation only: ETL + a few minutes of scoring. All three: 4x the ETL plus HPO per task |
| python | 3.12 via `uv sync` (§1) |

Nothing else travels to your site — no source CLIF data, no MIMIC data, only the model bundle.

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
| `<site>_baseline_phi/` | `cohort.parquet`, shared `_shared/MEDS/`, `features.npz`, `preds_<kind>.parquet` | No |
| `<site>_baseline_non_phi_for_upload/` | `codes.parquet`, `table1.json`, `report/<kind>/` JSONs, optional report PNGs | Yes |
| `<site>_baseline_models/` | Models this site fit (`<task>/local/`, `<task>/transfer/`) | No |
| `mimic_baseline_models/` | Public trained model bundle used for external validation | No |

`<kind>` is `external_validation`, `transfer`, or `local` — one per command you ran. Reports are kept side by side so the three can be compared; `codes.parquet` and `table1.json` describe your data rather than a model, so they are not duplicated per kind.

Quick upload check:

``` text
<site>_baseline_non_phi_for_upload/<task>/codes.parquet
<site>_baseline_non_phi_for_upload/<task>/table1.json
<site>_baseline_non_phi_for_upload/<task>/report/<kind>/overall.json        # episodic tasks
<site>_baseline_non_phi_for_upload/<task>/report/<kind>/landmark.json       # continuous tasks
<site>_baseline_non_phi_for_upload/<task>/report/<kind>/hospitalization_level.json
<site>_baseline_non_phi_for_upload/<task>/report/<kind>/viz/*.png
```

There is one `report/<kind>/` per command you ran — upload all of them. `codes.parquet` and `table1.json` sit above the kind folders and are not duplicated.

The upload folder should not contain `cohort.parquet`, `_shared/MEDS/`, `features.npz`, or any `preds_<kind>.parquet`.

## Common Site Commands

### Run All Tasks, Low-Memory Safe

This is the recommended default for most sites:

``` bash
uv run flair-baseline prepare --clif-config config/clif_config.json --out . --holdout-only --reuse --pmc
uv run flair-baseline external-validation --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
```

To lower memory further, reduce the batch size:

``` bash
uv run flair-baseline prepare --clif-config config/clif_config.json --out . --holdout-only --reuse --pmc --batch-size 1000
uv run flair-baseline external-validation --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
```

### Run All Tasks Without Batched ETL

Use this if the machine has enough memory and you want a simpler single-pass ETL:

``` bash
uv run flair-baseline prepare --clif-config config/clif_config.json --out . --holdout-only --reuse
uv run flair-baseline external-validation --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
```

### Run All Three Kinds

Prepare the full cohort once, then run the three model commands. They are independent — you can run them on different days, or stop after any of them.

``` bash
uv run flair-baseline prepare --clif-config config/clif_config.json --out . --reuse --pmc

uv run flair-baseline external-validation --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
uv run flair-baseline transfer-learning   --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
uv run flair-baseline local-training      --clif-config config/clif_config.json --out . --viz
```

Add `--no-hpo` to the two fitting commands for a much faster first pass.

### Run One Task Only

Add `--task icu_daily_mortality`, `--task icu_daily_ltach`, `--task extubation_failure_24h`, or `--task icu_readmission` to every command in the sequence.

``` bash
uv run flair-baseline prepare --clif-config config/clif_config.json --out . --holdout-only --reuse --pmc --task icu_daily_mortality
uv run flair-baseline external-validation --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz --task icu_daily_mortality
```

### Run The Stages Manually

Use this only if you want more control or need to restart from a specific stage:

``` bash
uv run flair-baseline build-cohorts --clif-config config/clif_config.json --out .
uv run flair-baseline build-data --clif-config config/clif_config.json --out . --holdout-only --reuse --pmc
uv run flair-baseline featurize --clif-config config/clif_config.json --out . --holdout-only
uv run flair-baseline external-validation --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
```

Drop `--holdout-only` from `build-data` and `featurize` if you intend to run `transfer-learning` or `local-training`.

## What The Pipeline Does

The workflow has four stages: three that prepare data, then one model command per kind.

| Stage | Command | Output |
|------------------------|------------------------|------------------------|
| 1 | `build-cohorts` | Per-task cohorts with train/test split and `table1.json` |
| 2 | `build-data` | One shared CLIF-to-MEDS store under `<site>_baseline_phi/_shared/MEDS/` |
| 3 | `featurize` | Per-task `features.npz` plus uploadable `codes.parquet` |
| 4 | `external-validation` / `transfer-learning` / `local-training` | Predictions, report JSONs, and optional visualizations |

The `prepare` command runs stages 1 through 3. Stage 4 can be run once per kind off the same prepared data.

At external sites, `--holdout-only` keeps the expensive ETL and featurization scoped to the 25% test split. The full cohort is still built so `table1.json` can describe the full local cohort.

## Tasks

| Task                     | Name                               | Report mode |
|----------------|-----------------------------------------|----------------|
| `icu_daily_mortality`    | ICU daily in-hospital mortality    | Continuous  |
| `icu_daily_ltach`        | ICU daily LTACH discharge          | Continuous  |
| `extubation_failure_24h` | Extubation failure within 24 hours | Episodic    |
| `icu_readmission`        | ICU readmission                    | Episodic    |

The report mode is not set here — each task declares it in its own `META`, and the baseline reads it from there. There are only two modes:

`continuous` scores multiple rows per stay, at lead-time landmarks. Its report carries a pooled cross-landmark aggregate plus the per-landmark curve behind it.

`episodic` scores one prediction per stay.

Each task's report is two JSONs (report schema 4):

| File | Mode | Contains |
|--------------------|--------------------|--------------------------------|
| `overall.json` | episodic | `discrimination` (AUROC/AUPRC + CIs), `calibration` |
| `landmark.json` | continuous | `pooled`, `by_landmark` |
| `hospitalization_level.json` | both | 99-threshold sweep, Youden, `fairness`, `net_benefit` |

Every report JSON stamps `metadata.report_schema`, so a reader can tell which bundle shape it is looking at. Earlier bundles (schema 3) split these across `discrimination.json`, `calibration.json`, `dca.json`, `fairness.json`, `operating_points.json`, `kpi.json` and `leadtime.json`. No metric was dropped in the consolidation — it moved into the files above.

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

The ELF config is the single source of truth for what is extracted into MEDS and what can become a feature.

Discharge diagnoses (`HOSP_DX`) are extracted into MEDS for audit but intentionally excluded from features, because they are assigned after the stay and leak outcome information.

## Feature Engineering {#feature-engineering}

Every feature summarizes the events of one code within the same stitched encounter (`hospitalization_join_id`) occurring **strictly before** the prediction time. The strict `<` is what makes the featurization leak-free — an event stamped exactly at `prediction_dttm` is invisible.

**Step 1 — code truncation.** ELF codes are `//`-delimited hierarchies of varying depth. Each is truncated by value type so that the *unit is never a feature*:

| code type | depth | example |
|------------------------|------------------------|------------------------|
| numeric (any event carries a `numeric_value`) | 2 levels, `DOMAIN//concept` | `LAB//lactate//mmol/l//bmp` → `LAB//lactate` |
| categorical (text only) | 3 levels, `DOMAIN//category//value` | `RESP//device_category//imv` (unchanged) |

Every lactate draw therefore feeds the same columns regardless of unit or order type.

**Step 2 — aggregation by role.** The truncated code determines how its events are summarized:

| role | applies to | columns | never observed |
|------------------|------------------|------------------|------------------|
| `ROLE_STAT` | labs, vitals, GCS/RASS, numeric respiratory parameters | 4: `code::min`, `::max`, `::mean`, `::median` over `numeric_value` | **`NaN`** — XGBoost routes it down its missing branch |
| `ROLE_COUNT` | `MED_CON//`, `MED_INT//` medications | 1: number of administration events | genuine `0` |
| `ROLE_ONEHOT` | `RESP//device_category//*`, `RESP//mode_category//*`, other categorical codes | 1: `1` if the code ever occurred before now, else `0` | `0` |

`NaN` versus `0` is load-bearing. A lab that was never drawn is not a lab whose value was zero, and conflating them is a real signal loss — clinicians order tests selectively, so "never measured" is itself informative. Medication counts are the opposite case: never given genuinely is zero exposure.

Medications are counted rather than summarized because the dose magnitude carries far less signal than the fact and frequency of exposure, and because MIMIC's `mar_action_category` values are all real administration events.

**Step 3 — dense extras.** Appended after the code columns, in fixed order so every site produces an identical layout:

| column | source | missing handling |
|------------------------|------------------------|------------------------|
| `age_at_admission` | cohort | filled with 0 |
| `sex__Female`, `sex__Male` | cohort | both `NaN` when sex is unknown/other |
| `time_since_first_hrs` | hours from the encounter's first event to the prediction | — |

**Result.** A dense `float32` matrix — on MIMIC, 442 code columns from 211 vocabulary codes, plus the 4 extras above for 446 total, identical across all four tasks. Dense rather than sparse because `tree_method="hist"` handles it better and a CSR would have to store every `NaN` explicitly.

**Vocabulary.** A fixed, committed `vocab.json` shared by all tasks and all sites, so every model has the same column space and bundles are interchangeable. Sites use the vocabulary shipped in the model bundle and never regenerate it.

## Model Owner Workflow

Use this section only if you are training or refreshing the source model bundle.

Set `config/clif_config.json` to the source training site and run:

``` bash
uv run flair-baseline build-cohorts --clif-config config/clif_config.json --out .
uv run flair-baseline build-data --clif-config config/clif_config.json --out .
uv run flair-baseline build-vocab --clif-config config/clif_config.json --out .
uv run flair-baseline featurize --clif-config config/clif_config.json --out .
uv run flair-baseline local-training --clif-config config/clif_config.json --out . --viz
```

The source site produces the `local` kind only — it has no shipped model to validate or transfer from. On MIMIC, `run_mimic_train.sh` wraps the last step: it fits all four tasks smallest-first, logs per-task wall-clock, and is resumable through per-task stamps in `.mimic_run_stamps/` (delete that folder to force a full retrain).

This writes:

``` text
<training_site>_baseline_phi/<task>/           cohort.parquet, features.npz, preds_local.parquet
<training_site>_baseline_non_phi_for_upload/<task>/   codes.parquet, table1.json, report/local/
<training_site>_baseline_models/<task>/local/  model.json, vocab.json, params.json
```

Publish only the model bundle folder. Sites point `--models-dir` at it; the commands resolve `<task>/local/` and fall back to a flat `<task>/` for bundles built before the kind split.

`build-vocab` is for maintainers/model owners. CLIF sites should use the vocabulary shipped in the model bundle and should not regenerate it.

Reference numbers from the MIMIC-IV source run (all four tasks, HPO on, 118 minutes total):

| task                   | test AUROC        |
|------------------------|-------------------|
| icu_daily_mortality    | 0.8673 *(pooled)* |
| icu_daily_ltach        | 0.7235 *(pooled)* |
| extubation_failure_24h | 0.7143            |
| icu_readmission        | 0.6705            |

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
uv run flair-baseline external-validation --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
```

### Need To Save Memory?

Keep `--pmc` (`poor-man's-compute`) on the `prepare` command. It batches the shared MEDS extraction into part files so peak RAM is bounded by one batch instead of the whole cohort.

The default `--pmc` batch size is `4000` encounters. If memory is still tight, set a smaller batch size:

``` bash
uv run flair-baseline prepare --clif-config config/clif_config.json --out . --holdout-only --reuse --pmc --batch-size 1000
```

Use a larger `--batch-size` only if the machine has enough memory.

You can also run one task at a time with `--task <task_name>`, using any name from the Tasks table above. A unique prefix works too, so `--task icu_read` resolves to `icu_readmission`.

## Development Notes

The baseline has its **own** uv environment. Run `uv sync` from this directory (not from the parent FLAIR repo) — that creates `flair_baseline/.venv`, and `uv run` uses it automatically.

The FLAIR benchmark library is bundled as a wheel under `wheels/` and pinned by path in `pyproject.toml`.

### Rebuilding The Bundled Wheel

Run this whenever `flair_benchmark` changes. The version stays `0.0.1`, so the rebuilt wheel has an identical filename — but `uv.lock` records the wheel's **sha256**, so the new content will not install until the lock is refreshed. You will see a hard `Hash mismatch` error rather than a silent stale install, so do not skip the `uv lock` step:

``` bash
# from the FLAIR repo root
rm -f flair_baseline/wheels/*.whl
uv build --wheel --out-dir flair_baseline/wheels
rm -f flair_baseline/wheels/.gitignore   # uv writes one; the wheel MUST be committed

# from flair_baseline/
uv lock --upgrade-package flair-benchmark   # re-pin the new wheel hash
uv sync --reinstall-package flair-benchmark
uv run pytest                            # tests/test_benchmark_contract.py is the tripwire
```

Commit `uv.lock` alongside the rebuilt wheel — the recorded hash is what makes a mismatched pair fail loudly instead of drifting.

`tests/test_benchmark_contract.py` exists for exactly this moment. It asserts that every task's report mode is one the benchmark still accepts, and that the headline-AUROC reader matches the current report bundle — so an incompatible rebuild fails in seconds instead of hours into a training run.

If the package version or wheel filename ever does change, update `[tool.uv.sources]` in `pyproject.toml` to match.

### After A Benchmark Upgrade, Do Not Reuse The Shared MEDS

`build-data --reuse` decides whether to skip the ETL by hashing its *inputs* (cohort membership, ELF domains, scope, batch size). It does **not** fingerprint the benchmark version, so a store built by an older library looks reusable even when its schema has changed — and the count join would then match zero rows without raising.

After rebuilding the wheel, delete `<site>_baseline_phi/_shared/` and re-run `build-data` without `--reuse`. `featurize` will refuse to write an all-zero feature matrix if you forget, but starting clean is cheaper than diagnosing it.