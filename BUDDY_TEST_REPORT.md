# Buddy Test Report

|                              |             |
|------------------------------|-------------|
| **Buddy site / institution** | *site name* |
| **Tester**                   | *name*      |
| **Date**                     | 2026-08-03  |

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
| 1 | Environment reproduces (`uv sync` / `00_renv_restore.R`, nothing by hand) | **Pass** | `uv sync --locked` succeeded in a clean clone and installed the project from the committed lockfile; `uv run flair-baseline --help` succeeded without manual installation. |
| 2 | Configuration works from `config/README.md` alone; no hardcoding | **Pass** | This project keeps the equivalent configuration instructions and parameter table in the root `README.md`, not a separate `config/README.md`. The Rush run used only the documented site config fields; no Rush path, site, or date range is hardcoded in `src/`. |
| 3 | Required tables/fields match what the code reads (mCIDE-valid) | **Pass** | The required CLIF tables documented in `README.md` match the cohort and ELF loaders. The Rush data passed clifpy validation and produced all four task cohorts and feature sets using mCIDE categories. |
| 4 | Runs end to end with no manual edits between steps | **Pass** | The Rush run completed `prepare`, external validation, transfer learning, and local training for all four tasks. All three prediction/report kinds are present for every task. Artifact timestamps span approximately 10 hours; no memory failure is recorded on the 16 GB host. |
| 5 | Outputs in `output/final_no_phi/` with right naming/type, no raw dumps | **Pass** | The project-specific equivalents are `rush_baseline_non_phi_for_upload/` and `rush_baseline_models/`. The non-PHI root contains only aggregate JSON reports, figures, and four code registries; the model root contains eight local/transfer model bundles. Row-level cohorts, features, MEDS events, and predictions remain under `rush_baseline_phi/`. |
| 6 | **Data security**: no PHI, every stat n \>= 10, no raw data *(blocking)* | **Pass** | Audited all 98 non-PHI files and 24 model files. No `patient_id` or `hospitalization_id` appears in paths, JSON, schemas, or model files. The only shareable Parquet files are aggregate code registries, each with minimum `event_count` 10. No unsuppressed positive count from 1 through 9 was found. |
| 7 | Clinical sanity: aggregates plausible for the cohort | **Pass** | Cohorts and outcome prevalence are plausible: mortality 53,476/10.3%, LTACH 53,476/3.0%, extubation failure 7,311/6.0%, and ICU readmission 29,755/7.9%. |
| 8 | Documentation usable: could run from the README alone | **Pass** | The root `README.md` provides setup, config fields, required tables, full-cohort preparation, all three external-site model commands, tasks, output locations, and sharing rules. No additional implementation document is required to run the workflow. |

## Overall verdict

**Verdict:** **Pass with notes**

- **Pass** - runs cleanly, output sane and secure, docs followable. Ready to distribute.
- **Pass with notes** - works; address the notes below.
- **Fail** - at least one blocking issue (doesn't run, output wrong/insecure, docs unfollowable). A data-security failure (check 6) is always a Fail.

### Blocking issues (must fix before distribution)

None.

### Non-blocking notes / suggestions

1.  Consider adding `config/README.md` or updating the generic buddy checklist to point to the root README's configuration section.
2.  Update `flow.md` sharing language to match `README.md`: external sites share both the generated non-PHI folder and generated models folder.