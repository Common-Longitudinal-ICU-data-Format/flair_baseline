# Buddy Test Report

|                              |             |
|------------------------------|-------------|
| **Buddy site / institution** | *site name* |
| **Tester**                   | *name*      |
| **Date**                     | 2026-08-07  |

## Environment

|            |              |
|------------|--------------|
| **OS**     | macOS 26.5.2 |
| **RAM**    | 16 GB        |
| **Python** | 3.12.11      |

The checks below use this project's documented output convention: `<site>_baseline_non_phi_for_upload/` and `<site>_baseline_models/`.

## Checks

| \# | Check | Result | Notes |
|:--------------:|------------------|---------------------|------------------|
| 1 | Environment reproduces (`uv sync` / `00_renv_restore.R`, nothing by hand) | **Pass** | Replaced and reinstalled the vendored `flair-benchmark` wheel, refreshed its lockfile hash, and verified `uv sync --locked` plus `uv run flair-baseline --help` in a clean checkout without manual installation. |
| 2 | Configuration works from `config/README.md` alone; no hardcoding | **Pass** | This project keeps the equivalent configuration instructions and parameter table in the root `README.md`, not a separate `config/README.md`. The Rush run used only the documented site config fields; no Rush path, site, or date range is hardcoded in `src/`. |
| 3 | Required tables/fields match what the code reads (mCIDE-valid) | **Pass** | The required CLIF tables documented in `README.md` match the cohort and ELF loaders. The Rush data passed clifpy validation and produced all four task cohorts and feature sets using mCIDE categories. |
| 4 | Runs end to end with no manual edits between steps | **Pass** | After deleting the old Rush outputs and cache, the clean Rush run completed `prepare`, external validation, transfer learning, and local training for all four tasks. Transfer and local training each completed 30 HPO trials per task. All three prediction/report kinds are present for every task, and no memory failure occurred on the 16 GB host. |
| 5 | Outputs in `output/final_no_phi/` with right naming/type, no raw dumps | **Pass** | The project-specific equivalents are `rush_baseline_non_phi_for_upload/` and `rush_baseline_models/`. The non-PHI root contains only aggregate JSON reports, figures, and four code registries; the model root contains eight local/transfer model bundles. Row-level cohorts, features, MEDS events, and predictions remain under `rush_baseline_phi/`. |
| 6 | **Data security**: no PHI, every stat n \>= 10, no raw data *(blocking)* | **Pass** | Audited all 98 non-PHI files and 24 model files. No row identifier appears in paths, JSON, schemas, or model files. The only shareable Parquet files are four aggregate code registries, each with minimum `event_count` 10. No unsuppressed positive count from 1 through 9 was found. |
| 7 | Clinical sanity: aggregates plausible for the cohort | **Pass with notes** | Fresh cohorts and outcome prevalence are: mortality 227,222/16.0%, LTACH 227,222/10.3%, extubation failure 7,311/6.0%, and ICU readmission 29,755/7.9%. Ages range from 18 to 123. The two daily tasks emit the temporal-overlap warning described below. |
| 8 | Documentation usable: could run from the README alone | **Pass** | The root `README.md` provides setup, config fields, required tables, full-cohort preparation, all three external-site model commands, tasks, output locations, and sharing rules. No additional implementation document is required to run the workflow. |

## Overall verdict

**Verdict:** **Pass with notes**

- **Pass** - runs cleanly, output sane and secure, docs followable. Ready to distribute.
- **Pass with notes** - works; address the notes below.
- **Fail** - at least one blocking issue (doesn't run, output wrong/insecure, docs unfollowable). A data-security failure (check 6) is always a Fail.

### Blocking issues (must fix before distribution)

None.

### Non-blocking notes / suggestions

1.  Review the daily-task temporal-overlap warning before distribution: 585 of 171,227 train rows (0.3%) occur at or after the first test prediction, and the latest is 126 days later. The replacement wheel uses `SPLIT_VERSION = 3` at stitched-encounter grain, and no encounter crosses splits.
2.  Review the daily-task overfit gap: row-level train AUROC is 1.0 for local and transfer models, versus held-out AUROC 0.8145-0.8677.
3.  Confirm whether ages 120-123 are valid source values or expected top-coding.
4.  Consider adding `config/README.md` or updating the generic buddy checklist to point to the root README's configuration section.
