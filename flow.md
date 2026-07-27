# flair-baseline — end-to-end flow

XGBoost **point-in-time feature** baseline for the FLAIR ICU tasks, trained on MIMIC-IV with a leak-free **75/25 split on `hospitalization_join_id`**. Every feature summarizes the ELF events of one code that occur **strictly before** each prediction's `prediction_dttm` — numeric codes as min/max/mean/median, medications as exposure counts, categorical respiratory settings as presence indicators (§2③). This is a **self-contained sub-project**: the FLAIR library ships as a prebuilt wheel under `wheels/`; the baseline carries its own config (`config/`). Train once at a source site, ship the **models folder** to other sites, re-run there — the code travels to the data.

Every run partitions its artifacts into three **site-prefixed** folders (see §3); only the non-PHI folder leaves the site.

```         
                         ┌─────────── prepare ───────────┐┌──── local-training │ transfer-learning │ external-validation ────┐
 task.build()            │                               ││                                                                  │
 (flair_benchmark) ──► cohort ──► FE-meds ──► MEDS.parquet ──► role featurizer ──► XGBoost ──► preds ──► report (+viz)
                       (split)   (CLIF→MEDS)  + codes.parquet   (dense, leak-free)  (model)   _<kind>   <kind>/ JSON / PNG
```

| stage | code | output (folder) |
|------------------------|------------------------|------------------------|
| ① cohort | `flair_benchmark/tasks/taskN_*.py` + `_split.py` | `cohort.parquet` (PHI) |
| ② FE-meds | `flair_benchmark/features/fe_meds.py` + `flair_benchmark/meds_etl/` | `MEDS.parquet` (PHI), `codes.parquet` (non-PHI) |
| ③ featurize | `flair_baseline/featurize.py` | in-memory sparse CSR |
| ④ train/score | `flair_baseline/train.py` | `<kind>/model.json`, `<kind>/vocab.json` (models), `preds_<kind>.parquet` (PHI) |
| ⑤ report | `flair_benchmark.report.build_report` | `report/<kind>/*.json` (+ `viz/*.png`) (non-PHI) |

Folder routing is owned by `flair_baseline/layout.py` (`TaskPaths`, `report_mode`).

------------------------------------------------------------------------

## 0. Setup (uv, self-contained)

``` bash
git clone <baseline-repo> flair_baseline && cd flair_baseline
git clone <flair-repo> flair          # FLAIR lib cloned in beside the baseline
uv sync                               # resolves flair-benchmark from ./flair (editable) + xgboost + matplotlib

cp config/clif_config.template.json config/clif_config.json   # set "site" + "data_directory"
```

`pyproject.toml` declares `flair-benchmark` as a path source (`{ path = "flair", editable = true }`). Developing inside the FLAIR repo? `ln -s .. flair` instead of cloning. Python 3.12.

------------------------------------------------------------------------

## 1. Commands

The pipeline is **staged**: data preparation runs once, then any number of model
commands run off the prepared artifacts. Every command emits the three
site-prefixed folders (§3); the `site` comes from the clif config and prefixes
every folder.

### Preparation (run once per site)

```         
flair-baseline build-cohorts --clif-config config/clif_config.json --out .
flair-baseline build-data    --clif-config config/clif_config.json --out . [--holdout-only] [--reuse] [--pmc]
flair-baseline featurize     --clif-config config/clif_config.json --out . [--holdout-only]

flair-baseline prepare       --clif-config config/clif_config.json --out .   # all three in one go
```

| flag | default | meaning |
|------------------------|------------------------|------------------------|
| `--clif-config` | `config/clif_config.template.json` | `clif_config.json` (path, filetype, timezone, `site`). `site=mimic` ⇒ 75/25 split |
| `--elf-config` | `flair_elf_config.yaml` | event concepts to extract (= the feature set) |
| `--out` | `.` | root for the three `<site>_baseline_*` folders |
| `--task` | all tasks | task name or prefix (`extubation` → `extubation_failure_24h`) |
| `--train-end` / `--test-start` | none | date cutoff for non-mimic sites (omit on mimic) |
| `--holdout-only` / `--full-cohort` | full-cohort | scope the ETL + featurization to the 25% test split |
| `--reuse` / `--no-reuse` | no-reuse | reuse the shared MEDS when its manifest key matches |
| `--pmc` / `--batch-size` | off / 4000 | batch the ETL for bounded RAM |

`--holdout-only` is the cheap path for a site that **only** wants
`external-validation`. The two fitting commands below need the train split, so a
site running all three must prepare the full cohort.

### Model commands — one per kind

The `kind` (`local`, `transfer`, `external_validation`) is the one axis that
separates a site's answers. It routes the preds file, the report directory and
the model bundle, and nothing else.

| command | fits? | base model | rows scored | writes |
|---|---|---|---|---|
| `local-training` | yes, on the local train split | — | all | `report/local/`, `models/<task>/local/` |
| `transfer-learning` | yes, continues the shipped ensemble | `--models-dir` | all | `report/transfer/`, `models/<task>/transfer/` |
| `external-validation` | no — frozen | `--models-dir` | test (or all, if prepared) | `report/external_validation/` |

All three take `--report/--no-report` and `--viz/--no-viz`; the two fitting
commands also take `--hpo/--no-hpo` and `--hpo-trials` (default 30).
`train` and `infer` remain as hidden aliases for `local-training` and
`external-validation`, so pre-existing scripts keep working.

``` bash
# source site (MIMIC): fit and ship
uv run flair-baseline local-training --clif-config config/clif_config_mimic.json --out . --viz

# external site: all three answers on the SAME local 25% holdout
uv run flair-baseline external-validation --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
uv run flair-baseline transfer-learning   --models-dir mimic_baseline_models --clif-config config/clif_config.json --out . --viz
uv run flair-baseline local-training      --clif-config config/clif_config.json --out . --viz
```

Copy a source site's `*_baseline_models/` to the new site. Each command rebuilds
nothing — it reads the prepared cohort + features, re-applies the deterministic
**75/25 block split**, and reports on the resulting **25% test** set. Send back
only `<site>_baseline_non_phi_for_upload/`. Because ELF standardizes the code
vocabulary across CLIF sites, a shipped model applies without code remapping.

**How transfer learning works.** XGBoost has no fine-tuning, so
`transfer-learning` keeps the shipped ensemble and appends trees fit on local
train rows (`xgb_model=`). Hyperparameters are tuned the same way as a
from-scratch fit — `xgb.cv` accepts no `xgb_model=`, so the search expresses
continued boosting as the base ensemble's raw output used as a per-row
`base_margin`, which is mathematically the same thing. The saved model is
self-contained (base trees + local trees); scoring it needs no companion file.
It resolves the shipped **vocab.json** rather than the local one, because the
base trees index into the source site's column layout.

------------------------------------------------------------------------

## 2. Stage detail

### ① Cohort — `task.build()` + `_split.py`

`taskN_*.build(clif_config, train_end, test_start) -> pl.DataFrame`, one row per `prediction_id`:

```         
hospitalization_id, hospitalization_join_id, prediction_dttm,
<label_column>, split, age_at_admission, sex_category, race_category,
ethnicity_category, window_number, prediction_id
```

- `hospitalization_join_id` = stitched-encounter key (clifpy `stitch_encounters`, 6 h gap; one block can span several hospitalizations).
- **Split** (`flair_benchmark/_split.py`, `assign_split`): on MIMIC the split is decided at the **block grain** — 75 % of distinct `hospitalization_join_id` → `train`, 25 % → `test`, broadcast to all rows. A block never spans both splits (no split leakage). `infer` re-applies this same deterministic 75/25 block split at any site (ordered by `prediction_dttm`) so external evaluation runs on a **25% holdout**.
- `prediction_id = {join_id}-{hospitalization_id}-{patient_id}-{window_number}`.

Written to `<site>_baseline_phi/<task>/cohort.parquet`.

### ② FE-meds — `fe_meds.build_meds_tables()` + `meds_etl/`

Runs the **vendored CLIF→MEDS ETL** over the cohort's `hospitalization_id`s for the domains listed in `flair_elf_config.yaml`, attaches `hospitalization_join_id`, and writes two MEDS-standard tables.

`data/MEDS.parquet` — one row per clinical event:

```         
subject_id (=patient_id), time, code, numeric_value, text_value,
hospitalization_id, hospitalization_join_id
```

ELF code format (`//` separates hierarchy levels):

```         
VITAL//heart_rate                         LAB//lactate//mmol/l//bmp
MED_CON//norepinephrine//mcg/kg/min//start    MED_INT//vancomycin//mg//given
RESP//device_category//imv   PA//gcs_total   PA//rass
HOSP_DX//ICD10CM//A41.9
```

`metadata/codes.parquet` — code registry: `code, description, parent_codes, concept_version, event_count, is_numeric_value, is_text_value` (counts \< 10 floored to 10 for PHI).

Notes: - **Timestamps**: clifpy loads each table converting UTC → site-local **once**, then `strip_tz` drops the tz (`replace_time_zone(None)`) — wall-clock preserved, no second conversion. - The ETL emit is **vectorized polars** (one frame per concept), not row-by-row. - The data table is **not** stamped with `prediction_id`/`prediction_dttm` — the prediction join and the leak filter live in the featurizer, keeping this table compact and reusable.

### ③ Role featurizer — `flair_baseline/featurize.py`

`compute_counts(events, task_df, …)` → `Counts`; `counts_to_X(counts, vocab, roles)` → `(X, prediction_ids, feature_names)`.

```         
feature[prediction_id, code] = aggregate of that code's events in the same
                               hospitalization_join_id with time < prediction_dttm
                                                              ↑ strict point-in-time
```

**Truncation first — the unit is never a feature.** ELF codes are `//`-hierarchies of varying depth, truncated by value type before aggregation:

| code type | depth | example |
|---|---|---|
| numeric (carries `numeric_value`) | 2, `DOMAIN//concept` | `LAB//lactate//mmol/l//bmp` → `LAB//lactate` |
| categorical (text only) | 3, `DOMAIN//category//value` | `RESP//device_category//imv` |

**Then aggregation by role** (`code_role()` — a pure function of the truncated string, so a committed `vocab.json` alone determines the column layout):

| role | applies to | columns | never observed |
|---|---|---|---|
| `ROLE_STAT` | labs, vitals, GCS/RASS, numeric resp params | 4 — `::min`, `::max`, `::mean`, `::median` | **NaN** (missing branch) |
| `ROLE_COUNT` | `MED_CON//`, `MED_INT//` | 1 — #administration events | genuine `0` |
| `ROLE_ONEHOT` | `RESP//device_category//*`, `RESP//mode_category//*`, other categoricals | 1 — ever-occurred 0/1 | `0` |

- **No leak**: events join predictions on `hospitalization_join_id`, then `time < prediction_dttm` (strict `<`). One feature row per `prediction_id`.
- **NaN ≠ 0**: a lab never drawn is not a lab that measured zero. Ordering is selective, so "never measured" is itself signal. Medication counts are the opposite — never given genuinely is 0.
- **Dense float32**: 442 columns from 211 codes on MIMIC. Even a 262k-row cohort stays under 500 MB, and `tree_method="hist"` suits dense better than a CSR that would store every NaN explicitly.
- **Scale**: truncated codes + encounter blocks are factorized to uint32 *before* the join (no strings in the \~100 M-row heavy path), and the join runs in `n_chunks` block-hash partitions so the exploded event×prediction frame peaks at \~1/`n_chunks`.
- **Vocabulary**: a fixed, committed `vocab.json` shared by every task and every site — not fit per run. `build-vocab` regenerates it (maintainer only).
- **HOSP_DX excluded**: `feature_exclude_prefixes: [HOSP_DX]` — discharge diagnoses are post-hoc (assigned for the whole stay at coding time) and leak. They stay in `MEDS.parquet` for audit but never become features.

### ④ XGBoost — `flair_baseline/train.py`

`train_and_score(X, prediction_ids, feature_names, task_df, label_col, …)`:

- Appends dense extras after the code columns, in a fixed code-side order so every site matches: `age_at_admission`, `sex__Female`, `sex__Male` (both NaN when sex is unknown/other), `time_since_first_hrs`.
- Hyperparameters come from an Optuna TPE search by default (30 trials, 5-fold `xgb.cv` AUROC on **train rows only**); `--no-hpo` falls back to a hand-tuned set. Fixed across both: `tree_method="hist"`, `random_state=0`.
- **`scale_pos_weight` is 1.0, deliberately.** Upweighting the rare positive class improves ranking but inflates predicted risk (O:E > 1) and wrecks calibration. Neutral weight keeps predictions on the prevalence scale; imbalance is reflected honestly in the probabilities instead.
- Fits on `split=="train"`, scores **every** row (so the report can bin by split).
- `base_model=` instead seeds the fit from a shipped booster (`xgb_model=`), appending locally-trained trees — this is what `transfer-learning` uses. `model_in=` skips fitting entirely and scores frozen.
- Writes `model.json` (booster), `vocab.json` (`{vocab, extras}`), and `preds.parquet`:

```         
prediction_id, hospitalization_id, hospitalization_join_id, split, <label_column>, y_prob, y_pred
```

### ⑤ Report — `flair_benchmark.report.build_report`

`build_report(preds, task_module, out, cohort_path=cohort, viz=…, site=…, mode=…)`. Demographics (`sex/race/ethnicity/age`) flow in from `cohort.parquet` for fairness. Each pillar is one JSON of metrics + raw curve arrays + **95% bootstrap CIs**, evaluated at the task's fixed clinical threshold. `--viz` renders PNGs from those same dicts (local sanity only).

The **mode** (chosen per task by `flair_baseline/layout.py:report_mode`) sets how a stay's many rows are scored:

| mode | tasks | files |
|---|---|---|
| **episodic** | extubation_failure_24h, icu_readmission | `discrimination.json`, `calibration.json`, `dca.json`, `fairness.json` |
| **landmark** | icu_daily_mortality, icu_daily_ltach | the four pillars **per lead-time landmark** + `leadtime.json` |
| **peak** | _(none currently)_ | `discrimination.json` + `fairness.json` (+ NNE); calibration/DCA omitted (peak-calibration trap) |

- **episodic** — one prediction per stay; the four pillars straight up.
- **landmark** — scores at 12 risk-set landmarks counting back from each stay's last window, so discrimination/calibration/DCA/fairness are reported *per lead-time*. `leadtime.json` adds per-window detection sensitivity, median lead-time (+CI), and the **predicted-risk trajectory by outcome** (mean risk for ever-positive vs negative stays at each landmark).
- **peak** — collapses each stay to its peak risk (a screening view); calibration/DCA are dropped because the max-of-many-windows is inflated and not honestly calibratable.

Why split this way: for a continuous / per-window model each stay contributes many correlated rows, so plain row-level metrics overstate deployed utility. Landmark (per lead-time) and peak (screening) are the two honest reductions; episode tasks have one row per stay already.

------------------------------------------------------------------------

## 3. Output tree (three site-prefixed folders)

`<site>` is the `site` field from the clif config (slugified). Only the non-PHI folder
is meant to leave the site; models are what `local-training` ships to other sites.

Artifacts that describe a **model** carry a `<kind>` segment; artifacts that describe
the site's **data** do not, and are shared across all three kinds rather than
duplicated. `<kind>` is one of `local`, `transfer`, `external_validation`.

```         
<out>/
  <site>_baseline_phi/                        ← stays local (PHI)
    _shared/MEDS/part-*.parquet   # ONE shared event store over the union of all cohorts
    <task_name>/
      cohort.parquet              # one row per prediction_id (+ split, label, demographics)
      features.npz                # cached feature-join (+ features_meta.json sidecar)
      preds_<kind>.parquet        # prediction_id, …, split, label, y_prob, y_pred

  <site>_baseline_non_phi_for_upload/<task_name>/   ← upload this
    codes.parquet             # code registry (description, version, counts<10 floored, value flags)
    table1.json               # cohort characteristics by split & label
    report/<kind>/
      overall.json            # headline discrimination/calibration/DCA/fairness (episodic)
      landmark.json           # the same, per lead-time landmark + pooled       (continuous)
      hospitalization_level.json
      viz/                    # only with --viz; PNGs rendered from the JSONs (local sanity)

  <site>_baseline_models/<task_name>/<kind>/  ← ship <kind>=local to other sites
    model.json                # XGBoost booster (transfer: base trees + local trees)
    vocab.json                # {vocab:[…codes…], roles:[…], features:[…], extras:[…]}
    params.json               # the tuned hyperparameters that produced it
```

A site holds one `report/<kind>/` per answer it computed. The source site (MIMIC)
computes `local` only; an external site can compute all three and compare them on
the same 25% holdout. Model bundles shipped before the `<kind>` split were flat
(`<task>/model.json`) and are still resolved.

Landmark tasks (1, 2) wrap each pillar as `{metadata, landmarks:[…]}` (one entry per
lead-time) and add `leadtime.json`. A peak-mode task would omit `calibration.json`/`dca.json`.

------------------------------------------------------------------------

## 4. What's used

### Feature domains — `flair_elf_config.yaml`

| domain | what | as feature? |
|------------------------|------------------------|------------------------|
| VITAL | all vital signs | ✅ |
| LAB | all labs | ✅ |
| RESP | respiratory-support settings | ✅ |
| PA | patient assessments — **GCS + RASS only** | ✅ |
| MED_CON | continuous meds (vasopressors, sedatives, …) | ✅ |
| MED_INT | intermittent meds (antibiotics, …) | ✅ |
| HOSP_DX | hospital discharge diagnoses | ❌ extracted, **excluded** (post-hoc/leaky) |

`PATIENT` / `HOSP` load implicitly for subject + hospitalization mapping (not featurized).

### Rules

- **Split**: 75/25 on `hospitalization_join_id` (mimic); a block never spans splits.
- **No leak**: features use only events with `time < prediction_dttm` (strict).
- **Vocabulary**: fit on train split only.
- **HOSP_DX excluded**: post-hoc; see §2③.
- **Timestamps**: UTC → site-local once (clifpy), then tz stripped; no second conversion.

------------------------------------------------------------------------

## 5. Per-task breakdown (MIMIC-IV)

### icu_daily_mortality — ICU Daily In-Hospital Mortality (7 AM)

- `label_mortality` · outcome · one-per-window (daily 07:00 grid) · threshold 0.95.
- Cohort **198,125** rows (148,763 train / 49,362 test; 9,838 test positives).
- MEDS events / codes per-domain: VITAL 44.999M, LAB 17.584M, RESP 11.726M, PA 8.282M, MED_CON 6.224M, MED_INT 2.681M, HOSP_DX 1.174M.
- Features **198,125 × 433**, 14.66M nnz. **Test AUROC 0.790**. 9 PNGs (+ encounter lead_time / operating_curve). Encounter: 16,102 stays (2,135 event); at t=0.95 stay-sens 0.017, ctrl-alarm 0.001, NNA 3.58 — lead ≈0 (daily mortality label fires on the event day).

### icu_daily_ltach — ICU Daily LTACH Discharge (7 AM)

- `label_ltach` · outcome · one-per-window (daily 07:00) · threshold 0.5.
- Cohort **198,125** (148,763 / 49,362; 7,152 test pos). Same base ICU cohort + events as icu_daily_mortality.
- Features **198,125 × 433**, 14.66M nnz. **Test AUROC 0.757**. 9 PNGs (+ encounter).

### extubation_failure_24h — Extubation failure within 24 h

- `label_extubation_failure_24h` · intervention · one-per-stay (episode) · threshold 0.15.
- Cohort **26,771** (20,077 / 6,694; 364 test pos).
- MEDS **57,832,570** events / 13,543 codes.
- Features **26,771 × 412**, 2.21M nnz. **Test AUROC 0.701**. 5 PNGs (one-per-stay → no window plots; `encounter.json` = `one_per_stay_task` stub, no lead-time/operating-curve PNGs).

### icu_readmission — Unplanned ICU Readmission

- `label_icu_readmission` · outcome · one-per-stay (episode) · threshold 0.2.
- Cohort **67,250** (50,436 / 16,814; 1,562 test pos).
- MEDS **78,185,174** events / 17,985 codes.
- Features **67,250 × 429**. **Test AUROC 0.645**. 5 PNGs (one-per-stay → `encounter.json` stub).

### Results summary

MIMIC-IV source run, 2026-07-27 — all four tasks, HPO on, 118 min total:

| task                         | test AUROC | features | n_test (pos)      |
|------------------------------|-----------:|---------:|-------------------|
| icu_daily_mortality    |  **0.867** |      446 | 65,659 (13,169)   |
| icu_daily_ltach        |  **0.724** |      446 | 65,659 (10,351)   |
| extubation_failure_24h |  **0.714** |      446 | 5,254 (284)       |
| icu_readmission        |  **0.670** |      446 | 13,256 (1,190)    |

The two continuous tasks report the pair-weighted pooled cross-landmark aggregate; the two episodic tasks report `discrimination.auroc`. Feature width is now identical across tasks (442 code columns from 211 vocabulary codes + 4 dense extras) because the vocabulary is a fixed committed file rather than fit per task.

Excluding HOSP_DX moved AUROC by ≤ 0.02 vs the leaky version — the post-hoc diagnoses were not real predictive signal, just leakage (and they had ballooned the vocab, e.g. icu_daily_mortality 1,840 → 433 under the old per-task vocabulary).

------------------------------------------------------------------------

## 6. Report catalog

| file | shows | modes |
|------------------------------------|------------------------------------|---|
| `discrimination.json` | AUROC/AUPRC (+CIs), raw ROC/PR curves, PR `baseline`=prevalence + `auprc_lift`, operating point at the clinical threshold (direction-aware action block; NNE in peak) | all |
| `calibration.json` | reliability bins, ECE, Brier, logistic slope/intercept, O:E ratio | episodic, landmark |
| `dca.json` | decision-curve net benefit vs threshold (+ clinical operating point; complement transform for `clinical_direction: below`) | episodic, landmark |
| `fairness.json` | per-subgroup metrics + parity gaps at the operating threshold (pinned subgroup set; subgroups n \< 10 dropped) | all |
| `leadtime.json` | per-window detection sensitivity (+CI), median lead-time (+CI), `trajectory` = predicted risk by eventual outcome | landmark |

Landmark files wrap each pillar as `{metadata, landmarks:[…]}` (one entry per lead-time).

### Viz catalog (only with `--viz`; local sanity, rendered from the JSON)

| png | mode | what it plots |
|---|---|---|
| `roc_pr.png` | all | ROC (AUROC) + PR with the prevalence baseline (AUPRC + lift) |
| `calibration.png` | episodic | reliability points + logistic recalibration; title = slope/intercept/O:E |
| `dca.png` | episodic | net benefit vs threshold, model vs treat-all/treat-none |
| `fairness_auroc.png` | all | AUROC per demographic subgroup (sex/race/ethnicity/age) |
| `landmark_skill.png`, `landmark_sensitivity.png`, `landmark_calibration.png` | landmark | metric vs windows-before-end, per-point `n` annotated |
| `landmark_risk_trajectory.png` | landmark | mean predicted risk vs windows-before-end, positive (label=1) vs negative (label=0) |

**Calibration caveat (rare-event tasks):** `scale_pos_weight` inflates predicted probabilities for the rare positive class — the model **ranks** well (AUROC) but is not calibrated (slope ≪ 1, predicted ≫ observed). This is why a very-low-prevalence task belongs in **peak** mode (calibration/DCA omitted). For tasks that do report calibration, recalibrate (Platt/isotonic) on a held-out split → slope/ECE/DCA become meaningful and cross-site comparable.

------------------------------------------------------------------------

## 7. Scale & memory

- Featurizer factorizes codes + blocks to int32 and runs the event→prediction join in `n_chunks` (default 8) block-hash partitions → peak ≈ 1/`n_chunks` of the aggregation. Even a multi-million-row hourly grid (\~95M events, \~932M nnz) fits a 16 GB box (\~12 GB peak).
- Everything downstream of the ELF events is sparse CSR.
- `--reuse` skips the ETL (reads existing `cohort.parquet` + `data/MEDS.parquet`) — re-featurize / retrain / re-report in minutes (seconds for the small tasks).
- Full cold sweep ≈ ETL (a few min per task; loads full CLIF tables once).

------------------------------------------------------------------------

## 8. Correctness guarantees (and how to re-check)

| guarantee | check |
|------------------------------------|------------------------------------|
| no leak | every counted event has `time < prediction_dttm` (strict, in `featurize.py`); sum of CSR counts == #events before each `prediction_dttm` |
| split disjoint | `train` vs `test` `hospitalization_join_id` sets disjoint, block ratio = 0.7500 |
| vocab train-only | vocab built from `split=='train'` predictions only |
| HOSP_DX excluded | `HOSP_DX=0` in every `vocab.json`, but present in `data/MEDS.parquet` |
| timestamps | tz stripped after one UTC→local convert; all datetimes tz-naive |

------------------------------------------------------------------------

## 9. Standardized output & cross-site aggregation

FLAIR is federated: each site runs locally and shares only `<site>_baseline_non_phi_for_upload/`
(the report JSONs + Table 1 + codes registry). PHI (cohort, MEDS, preds) and PNGs never leave the
site. The report is built to be *pooled* without raw patient data:

**(1) JSON is the canonical, poolable artifact; PNGs are local sanity.** Every PNG is rendered
*from* its JSON dict, so the JSON alone suffices.

**(2) Each pillar carries raw curve arrays + per-cell counts + 95% bootstrap CIs**, so a coordinator
can meta-analyze: n-weighted ROC/PR curves, Σ observed/expected per calibration bin → pooled
reliability + ECE, n-weighted DCA net benefit, and (landmark) per-lead-time sensitivity + risk
trajectory. Prevalence-awareness is built in — AUPRC reported as **lift** over prevalence and PR
carries the prevalence `baseline` — so a 0.15%-prevalence site and a 20%-prevalence site stay
comparable.

**(3) PHI suppression:** any cell with n < 10 is suppressed (`"<10"`) and excluded from pools;
codes.parquet counts are floored at the same threshold.

Each JSON's `metadata` block stamps `task`, `task_script_sha256` (same hash ⇒ identical label logic),
`flair_version`, `report_mode`, `split`, `n_stays`, `prevalence`, and `evaled_at` (the operating
threshold) — enough for a coordinator to align and pool sites.

------------------------------------------------------------------------

## 10. External-site validation

``` bash
# copy a training site's models folder to this site, then:
uv run flair-baseline external-validation --task extubation_failure_24h \
  --models-dir mimic_baseline_models \
  --clif-config config/clif_config.json --out . --viz

# optional — the other two answers, on the same local holdout (needs a
# full-cohort prepare, i.e. WITHOUT --holdout-only):
uv run flair-baseline transfer-learning --task extubation_failure_24h \
  --models-dir mimic_baseline_models \
  --clif-config config/clif_config.json --out . --viz
uv run flair-baseline local-training --task extubation_failure_24h \
  --clif-config config/clif_config.json --out . --viz
```

`infer` rebuilds the cohort + features from the **local** data, re-applies the deterministic 75/25
block split, featurizes against the **saved vocabulary**, and scores the saved model on the resulting
**25% test** set — then writes this site's three folders. Because ELF standardizes the code vocabulary
across CLIF sites, the MIMIC-trained model applies without code remapping. Share back only
`<site>_baseline_non_phi_for_upload/`.