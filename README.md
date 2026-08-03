# FLAIR Baseline

Train the source XGBoost models on MIMIC, or evaluate them at an external CLIF site. Patient-level data stays at the site.

## Choose A Workflow

| Who | Goal | Data preparation | Commands after preparation |
|---|---|---|---|
| MIMIC model owner | Build the model bundle | Full cohort plus a new vocabulary | `local-training` via `run_mimic_train.sh` |
| External site | Compare all three approaches | Full cohort | `external-validation`, `transfer-learning`, `local-training` |

External sites must prepare the full cohort. Do not use `--holdout-only` because transfer learning and local training require the train split.

## One-Time Setup

Install the environment from the repository root:

```bash
uv sync
```

Create the config required by your role:

| Role | Command | Required `site` value |
|---|---|---|
| MIMIC model owner | `cp config/clif_config.template.json config/clif_config_mimic.json` | `mimic` |
| External CLIF site | `cp config/clif_config.template.json config/clif_config.json` | A short local site name |

Edit the copied file with the local CLIF data location. Config fields are listed in [Config Parameters](#config-parameters).

## MIMIC Model Owner

Build cohorts, create the shared data, generate the source vocabulary, featurize, and train all four models:

```bash
uv run flair-baseline build-cohorts \
  --clif-config config/clif_config_mimic.json \
  --out .

uv run flair-baseline build-data \
  --clif-config config/clif_config_mimic.json \
  --out .

uv run flair-baseline build-vocab \
  --clif-config config/clif_config_mimic.json \
  --out .

uv run flair-baseline featurize \
  --clif-config config/clif_config_mimic.json \
  --out .

./run_mimic_train.sh
```

The training script runs all tasks smallest-first and resumes from stamps in `.mimic_run_stamps/`. Delete that directory to retrain every task. Pass `--no-hpo` for a faster fixed-parameter run or `--hpo-trials N` to change the search budget. Do not pass `--task` to this wrapper; use `local-training --task NAME` directly for a single task.

Publish only this model bundle:

```text
mimic_baseline_models/
  <task>/local/model.json
  <task>/local/vocab.json
  <task>/local/params.json
```

`model.json` and `vocab.json` are required by external sites. `params.json` is included for auditability.

## External Sites

Copy the published `mimic_baseline_models/` folder into the repository root. Prepare the full cohort once, then run all three model commands:

```bash
uv run flair-baseline prepare \
  --clif-config config/clif_config.json \
  --out . \
  --reuse \
  --pmc

uv run flair-baseline external-validation \
  --models-dir mimic_baseline_models \
  --clif-config config/clif_config.json \
  --out .

uv run flair-baseline transfer-learning \
  --models-dir mimic_baseline_models \
  --clif-config config/clif_config.json \
  --out .

uv run flair-baseline local-training \
  --clif-config config/clif_config.json \
  --out .
```

The model commands are independent after `prepare`. Add `--no-hpo` to `transfer-learning` and `local-training` for a faster fixed-parameter run.

## Workflow Parameters

The recommended values for each workflow are:

| Parameter | MIMIC owner | External sites |
|---|---|---|
| `--clif-config` | `config/clif_config_mimic.json` | `config/clif_config.json` |
| `--out` | `.` | `.` |
| `--models-dir` | Not used | `mimic_baseline_models` for external validation and transfer learning |
| `--holdout-only` | No | No |
| `--reuse` | Optional on `build-data` | Recommended |
| `--pmc` | Optional on `build-data` | Recommended |
| `--task` | Omit for all tasks | Omit for all tasks |
| `--no-hpo` | Optional | Optional for transfer and local training |

### Shared CLI Parameters

| Parameter | Default | Commands | Meaning |
|---|---|---|---|
| `--clif-config PATH` | `config/clif_config.template.json` | All | CLIF site config file |
| `--out PATH` | `.` | All | Root directory for site-prefixed outputs |
| `--task NAME` | All tasks | All except `build-vocab` | Run one exact task name or unique task prefix |

### Data Preparation Parameters

| Parameter | Default | Commands | Meaning |
|---|---|---|---|
| `--elf-config PATH` | `flair_elf_config.yaml` | `prepare`, `build-data` | ELF domain configuration |
| `--train-end YYYY-MM-DD` | Unset | `prepare`, `build-cohorts` | Last date assigned to train; use with `--test-start` |
| `--test-start YYYY-MM-DD` | Unset | `prepare`, `build-cohorts` | First date assigned to test; use with `--train-end` |
| `--holdout-only` | Off | `prepare`, `build-data`, `featurize` | Process only test rows; not used by the required external-site workflow |
| `--full-cohort` | On | `prepare`, `build-data`, `featurize` | Process train and test rows |
| `--reuse` | Off | `prepare`, `build-data` | Reuse shared MEDS data when its manifest matches |
| `--no-reuse` | On | `prepare`, `build-data` | Rebuild shared MEDS data |
| `--pmc` | Off | `prepare`, `build-data` | Batch ETL to reduce peak memory |
| `--no-pmc` | On | `prepare`, `build-data` | Run ETL without batching |
| `--batch-size N` | `4000` | `prepare`, `build-data` | Encounters per batch when `--pmc` is enabled |
| `--vocab-out PATH` | `vocab.json` | `build-vocab` | Destination for the generated source vocabulary |

If both split dates are omitted, MIMIC uses a deterministic patient-level 75/25 split and external sites use an oldest/newest patient-level 75/25 split. Supply both date parameters when using explicit dates.

### Model Parameters

| Parameter | Default | Commands | Meaning |
|---|---|---|---|
| `--models-dir PATH` | Required | `external-validation`, `transfer-learning` | Published source model bundle |
| `--vocab PATH` | `vocab.json` | `local-training` | Fixed vocabulary used for a new local model |
| `--report` / `--no-report` | Report on | All model commands | Enable or disable report JSON generation |
| `--viz` / `--no-viz` | Visualizations on | All model commands | Enable or disable report PNG generation |
| `--hpo` / `--no-hpo` | HPO on | `local-training`, `transfer-learning` | Enable Optuna search or use fixed parameters |
| `--hpo-trials N` | `30` | `local-training`, `transfer-learning` | HPO trial budget |

## Tasks

Use one of these values with `--task`. Add the same `--task` value to `prepare` and every subsequent model command in an external-site workflow.

| Task | Outcome |
|---|---|
| `icu_daily_mortality` | ICU daily in-hospital mortality |
| `icu_daily_ltach` | ICU daily LTACH discharge |
| `extubation_failure_24h` | Extubation failure within 24 hours |
| `icu_readmission` | ICU readmission |

## Config Parameters

| Field | Required | Default | Meaning |
|---|---|---|---|
| `site` | Yes | None | Site label used in output folder names; use `mimic` for the source site |
| `data_directory` | Yes | None | Directory containing local `clif_*` files |
| `filetype` | Yes | None | `parquet` or `csv` |
| `timezone` | No | `US/Central` | Timezone used for CLIF datetimes |
| `stitch_time_interval_hours` | No | `6` | Maximum gap used for encounter stitching |
| `cache_directory` | No | `./output` in the template | Local CLIF cache directory; if omitted, uses `<data_directory>/_flair_cache` |

The default workflow expects these CLIF tables:

| Required CLIF tables |
|---|
| `patient`, `hospitalization`, `adt` |
| `vitals`, `labs`, `respiratory_support`, `patient_assessments` |
| `medication_admin_continuous`, `medication_admin_intermittent` |

Files are normally named `clif_<table>.parquet` or `clif_<table>.csv`. `hospital_diagnosis` is optional and enriches Table 1 when available.

## External Site Sharing

Share both generated folders:

```text
<site>_baseline_non_phi_for_upload/
<site>_baseline_models/
```

The non-PHI folder contains `codes.parquet`, `table1.json`, and one `report/<kind>/` directory for each model command. The models folder contains the generated local and transfer-learning models.

Do not upload:

```text
<site>_baseline_phi/
mimic_baseline_models/
```

The PHI folder contains cohorts, features, MEDS events, and predictions and must remain at the site.
