# Buddy Test Report — UCMC

|                              |                                             |
|------------------------------|---------------------------------------------|
| **Buddy site / institution** | UCMC (University of Chicago Medical Center) |
| **Tester**                   | *name*                                      |
| **Date**                     | 2026-08-05                                  |
| **Kit revision**             | `b908644`                                   |

## Environment

|              |                                        |
|--------------|----------------------------------------|
| **OS**       | macOS 26.5.2, Apple M2 Max, 64 GB      |
| **Python**   | 3.12.11 (`uv` 0.8.3)                   |
| **Key deps** | clifpy 0.5.0, xgboost 3.3.0, polars 1.42.1 |
| **Data**     | CLIF 2.1.0 parquet, `US/Central`, 6 h stitching |
| **Run**      | Full external-site workflow, single clean pass, 5 h 35 m |

## Checks

| \# | Check | Result | Notes |
|:--:|---|---|---|
| 1 | Environment reproduces (`uv sync`, nothing by hand) | **Pass** | `uv sync --locked` clean from the committed lockfile; no manual installation. |
| 2 | Configuration works from docs alone; no hardcoding | **Pass** |  No site name, path, or date hardcoded in `src/`. |
| 3 | Required tables/fields match what the code reads (mCIDE-valid) | **Pass** | UCMC's category values match the code's match lists against the clifpy 2.1 schema. |
| 4 | Runs end to end with no manual edits between steps | **Pass** |  All tasks produced. |
| 5 | Outputs in the right place, no raw dumps | **Pass** | Row-level data confined to `ucmc_baseline_phi/` |
| 6 | **Data security**: no PHI, every stat n ≥ 10, no raw data *(blocking)* | **Pass with notes** | No direct identifiers in 122 audited files. Suppression applied and stamped; no integer 1–9 printed. But it is reversible by differencing. |
| 7 | Clinical sanity: aggregates plausible for the cohort | **Pass w/ notes** | Split is patient-level (0 patients span train/test), timezone and point-in-time discipline verified,  outlier bounds applied once. Refer to suggestions below. |
| 8 | Documentation usable: could run from the README alone | **Pass** | Setup, config fields, required tables, all three model commands, output locations, and sharing rules are covered. |

## Overall verdict

**Verdict:** **Pass with notes**

Refer Non-blocking notes/ suggestions below

### Blocking issues (must fix before distribution)

NA

### Non-blocking notes / suggestions

**1. — Medication counts include non-administrations, on a site-dependent scale.**
Truncation to two levels drops `mar_action_category`, so `ROLE_COUNT` counts every row
regardless of action. `featurize.py:69-73` justifies this on a MIMIC-only observation that does
not hold here: UCMC's most common medication event is `verify` and  `not_given`. **42 of 118 `ROLE_COUNT`
columns are inflated >1.5×** — lidocaine 9.3×, furosemide 8.4×, norepinephrine 2.2×. Affects
`external_validation` and `transfer`; `local` is unaffected. Counting `not_given` as exposure is
wrong at every site including MIMIC.
*Fix:* filter to `mar_action_group == 'administered'` in the ETL, before truncation. Preserves
the column space; only counts change. MIMIC must retrain.

**2 — Vocabulary coverage is unmeasured, and absent codes are misencoded as zero.**
Codes outside `vocab.json` are silently dropped (`featurize.py:367-376`)  Largest single drop is
`RESP//device_category//room_air` at 547,890 events: a valid mCIDE category MIMIC never charts,
so the vocabulary has no column for it. Conversely **30 vocabulary codes have zero local events
— 29 `ROLE_COUNT`, 1 `ROLE_ONEHOT`, none `ROLE_STAT`** — so all 30 reach XGBoost as a genuine
`0.0` meaning "never given" rather than "not charted here". The `nnz == 0` guard catches
neither; it runs before vocabulary resolution.
*Fix:* rebuild `vocab.json` over the union of participating sites' code spaces; emit coverage
diagnostics into the non-PHI bundle; hard-fail below a floor.

**3 — LTACH external validation is near chance** (0.597 against local 0.760).  Plausibly genuine non-transportability or bias from the above two issues. 
