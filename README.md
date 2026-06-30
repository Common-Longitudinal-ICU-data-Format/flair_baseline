# flair-baseline

XGBoost count-feature baseline for the FLAIR ICU benchmark (all 5 tasks).

A **self-contained sub-project**: it carries its own config and pins the FLAIR library, bundled as a prebuilt wheel under `wheels/` (until `flair-benchmark` is published on PyPI). Train once at a source site (MIMIC), ship the **models folder** to other sites, and re-run there — the code travels to the data, not the other way around.

## What it does

The pipeline is **staged** — run the subcommands in sequence (the CLIF→MEDS ETL is built **once**, not per task):

1. **`build-cohorts`** → each task's cohort (one row per `prediction_id`, with `split`), always carrying both train+test. On MIMIC the split is **75/25 on `hospitalization_join_id`** (an encounter never spans train and test); at an external site the same deterministic 75/25 block split is re-applied, so the model is evaluated on a **25% holdout** of that site's data. Also writes `table1.json`.
2. **`build-data`** runs the vendored CLIF→MEDS ETL **once** over the **union of all task cohorts** — each `hospitalization_join_id` expanded to its full **stitched membership** (sibling hospitalizations included, so no encounter event is dropped) — into one shared `_shared/MEDS/` store.
3. **`featurize`** — per task, counts each ELF code with `time < prediction_dttm` (strict point-in-time, no leakage) off the shared store, and derives that task's `codes.parquet` (code registry, counts floored to the suppression threshold). The feature **vocabulary is a fixed, committed `vocab.json`** shared by every task and site (not fit per task) — so every model has the same columns and bundles are interchangeable.
4. **`train`** fits XGBoost per task on the fixed vocab; **`infer`** loads a shipped model instead. Both score all rows → `preds.parquet` + a **FLAIR report** bundle, scored by the per-task mode below.

### Report bundle (per task)

`build_report` is keyed on the unit of prediction. Each pillar is one JSON of metrics + raw curve arrays + 95% bootstrap CIs, evaluated at the task's fixed clinical threshold; `--viz` also renders sanity-check PNGs (never uploaded).

| mode | tasks | pillars |
|------------------------|------------------------|------------------------|
| **episodic** | task3, task5 | `discrimination`, `calibration`, `dca`, `fairness` |
| **landmark** | task1, task2 | the four pillars **per lead-time landmark** + `leadtime` (per-window sensitivity, median lead-time, risk trajectory by outcome) |
| **peak** | task4 | `discrimination` + `fairness` (+ NNE); calibration/DCA omitted (peak-calibration trap) |

Feature domains (`flair_elf_config.yaml`): all vitals, all labs, respiratory support, patient assessments (GCS + RASS), continuous + intermittent medications.

> The ELF config is the **single source of truth**: the domains listed there are exactly what gets extracted into MEDS *and* exactly what the model can use as features (no separate exclude list). `HOSP_DX` (discharge diagnoses, timestamped at discharge → leaky) is intentionally omitted. To include/exclude a domain, add/remove it there.

## Output: three site-prefixed folders

Every run writes under one `--out` root (default `.`), into three folders prefixed with the `site` from your config. Only the **non-PHI** folder is meant to leave the site.

```         
<site>_baseline_phi/_shared/MEDS/            shared CLIF→MEDS events (built once, all tasks)   ← stays local (PHI)
<site>_baseline_phi/<task>/                  cohort.parquet, features.npz, preds.parquet      ← stays local (PHI)
<site>_baseline_non_phi_for_upload/<task>/   codes.parquet, table1.json, report/*.json, report/viz/*.png   ← upload this
<site>_baseline_models/<task>/               model.json, vocab.json                            ← ship to other sites
```

Model folders keep their **training-site** prefix; PHI + non-PHI folders carry the **running-site** prefix.

## Two roles

| role | who | does | publishes |
|------------------|------------------|------------------|------------------|
| **Model owner** | the FLAIR baseline team (trains on MIMIC) | `train` → fits the 5 models | uploads `mimic_baseline_models/` to the FLAIR website |
| **CLIF site** | each participating ICU site | downloads the models, `infer` on local data | uploads its `*_non_phi_for_upload/` results back |

Only two things ever move: the **models folder** (owner → site) and the **non-PHI results folder** (site → FLAIR). Patient data never leaves a site.

### Shared setup (both roles, once)

``` bash
git clone <baseline-repo> flair_baseline && cd flair_baseline
uv sync                               # installs flair-benchmark from wheels/*.whl + xgboost

cp config/clif_config.template.json config/clif_config.json
# edit config/clif_config.json: set "site" (prefixes every output folder) and "data_directory"
```

The fixed feature vocabulary `vocab.json` ships committed in this repo — both `train` and
`infer` load it, so every model and every site share one column space. Maintainers
regenerate it with `build-vocab` (see below); sites never touch it.

For both train and inference sites, the FLAIR library ships **prebuilt** as `wheels/flair_benchmark-*.whl` (committed in this repo), so there is nothing to clone or symlink — `uv sync` installs it from there. This is the interim mechanism until `flair-benchmark` is published on PyPI, at which point the `[tool.uv.sources]` wheel pin drops and `uv sync` resolves it from the index.

> **Updating the bundled library** (maintainers only): rebuild the wheel from the FLAIR repo, then re-sync.
>
> ``` bash
> uv build --wheel --out-dir flair_baseline/wheels   # run from the flair repo
> rm -f flair_baseline/wheels/.gitignore             # uv writes a `*` ignore — drop it so the wheel ships
> # then, in flair_baseline/:
> rm uv.lock && uv lock                              # same version → uv pins the OLD hash; regen to pick up new wheel
> uv sync --reinstall-package flair-benchmark
> ```
>
> (If the version string changed, also bump the filename in `pyproject.toml` `[tool.uv.sources]`.)

------------------------------------------------------------------------

### Role A — Model owner (FLAIR baseline, on MIMIC)

1.  Set `"site": "mimic"` and the MIMIC `data_directory` in `config/clif_config.json`.

2.  Run the staged pipeline, then train all 5 tasks:

    ``` bash
    uv run flair-baseline build-cohorts --clif-config config/clif_config.json --out .
    uv run flair-baseline build-data    --clif-config config/clif_config.json --out .   # ETL once
    uv run flair-baseline featurize     --clif-config config/clif_config.json --out .
    uv run flair-baseline train         --clif-config config/clif_config.json --out . --viz
    # or, the first three in one go:  uv run flair-baseline prepare --clif-config … --out .
    ```

    > **Low-memory box?** add the hidden `--pmc --batch-size N` to `build-data` (poor-man's-compute):
    > it extracts the shared MEDS in sequential N-encounter batches to part-files, so peak RAM
    > is one batch instead of the whole cohort. Output is identical, just slower.

3.  Three folders appear: `mimic_baseline_phi/` (local, incl. the shared `_shared/MEDS/`), `mimic_baseline_non_phi_for_upload/` (MIMIC's own results), and **`mimic_baseline_models/`** — the per-task `model.json` + `vocab.json`.

4.  **Publish `mimic_baseline_models/`** to the FLAIR website as the downloadable model bundle. That folder is all a CLIF site needs.

> **Regenerating the fixed vocabulary** (maintainers, rarely): after `build-data`, run
> `uv run flair-baseline build-vocab --clif-config config/clif_config.json --out .` → writes
> `vocab.json`; commit it. All tasks/sites then share that feature space.

------------------------------------------------------------------------

### Role B — CLIF site (participating ICU)

1.  Finish the shared setup above; set your own `"site"` (e.g. `"rush"`) and `data_directory`.

2.  **Download the models folder** from the FLAIR website and drop it into the baseline folder, e.g. `flair_baseline/mimic_baseline_models/` (unzip here; keep the per-task subfolders).

3.  Run inference — same staged commands, but scope the data pull to your **25% holdout**
    with `--holdout-only` (only those encounters' join ids are ETL'd / featurized):

    ``` bash
    uv run flair-baseline build-cohorts --clif-config config/clif_config.json --out .
    uv run flair-baseline build-data    --clif-config config/clif_config.json --out . --holdout-only
    uv run flair-baseline featurize     --clif-config config/clif_config.json --out . --holdout-only
    uv run flair-baseline infer --models-dir mimic_baseline_models \
      --clif-config config/clif_config.json --out . --viz
    # one task only: add --task task1 to each command
    ```

    Your cohort is still built in full (train+test, so Table 1 ships complete) — only the
    ETL + featurization are restricted to the test-split encounters, saving compute.

4.  Three folders appear, prefixed with **your** site name:

    - `rush_baseline_phi/` — cohort, shared `_shared/MEDS/`, features, preds → **stays on your machine** (PHI, never upload).
    - `rush_baseline_models/` — the models you ran (already public).
    - **`rush_baseline_non_phi_for_upload/`** — codes registry, Table 1, report JSONs + PNGs.

5.  **Upload only `rush_baseline_non_phi_for_upload/`** to the FLAIR website (or the provided secure Box link). Nothing else leaves your site.

> Quick check before upload: the folder holds only `codes.parquet`, `table1.json`, and `report/` — no `cohort.parquet`, `_shared/MEDS/`, `features.npz`, or `preds.parquet`. All cell counts \< 10 are already suppressed (`"<10"`).

## Results (MIMIC-IV, 75/25 join-id split)

> Note: the table below predates the shared-MEDS rebuild. Pulling every stitched
> **sibling** hospitalization (previously dropped) adds events, so counts and AUROCs
> shift slightly upward on retrain — regenerate the table after a fresh `train`.

**Report AUROC** is the headline metric in `discrimination.json` — the one that gets uploaded. It is mode-matched (episodic = one row/stay; peak = stay-peak risk; landmark = the lead-0 landmark, i.e. each stay's last window). The **row-level** column is the raw per-prediction-row AUROC for context — for peak/landmark tasks it differs because a stay contributes many correlated rows, so it is not the deployed metric.

| task | mode | report AUROC (test) | n_stays | row-level |
|---------------|---------------|--------------:|--------------:|--------------:|
| task1 ICU daily mortality | landmark (lead-0) | 0.864 \[0.856, 0.872\] | 18,195 | 0.790 |
| task2 ICU daily LTACH | landmark (lead-0) | 0.789 \[0.775, 0.803\] | 18,195 | 0.752 |
| task3 extubation failure 24h | episodic | 0.696 \[0.665, 0.719\] | 6,693 | 0.696 |
| task4 sepsis ABX 6h | peak | 0.693 \[0.678, 0.709\] | 19,943 | 0.815 |
| task5 ICU readmission | episodic | 0.641 \[0.626, 0.658\] | 16,806 | 0.641 |

> Landmark tasks report 12 per-lead-time AUROCs (see `leadtime.json` / the `landmarks[]` array); the table shows lead-0 as a single summary. task4 peak (0.69) vs its row-level (0.81) is the peak-trap in action — per-window pooling overstates deployed skill.

## Scale notes

- The featurizer factorizes codes + encounter blocks to int32 and runs the event→prediction join in `n_chunks` block-hash partitions (default 8), so peak memory is \~1/n_chunks of the aggregation. This keeps even task4 — **14.3M hourly predictions, \~95M events, \~930M feature non-zeros** — within a 16 GB machine (peak ≈ 12 GB, \~21 min featurize + \~5 min train). Features are sparse CSR throughout.
- `build-data` runs the CLIF→MEDS ETL **once** over the union of all 5 task cohorts (each encounter expanded to its full stitched membership) → `_shared/MEDS/`. Since the tasks overlap heavily on `hospitalization_join_id`, this replaces up to 5× redundant per-task extraction. `featurize` scans the shared store lazily, restricting to each task's encounters via the inner join.
- The hidden `-pmc` (poor-man's-compute) mode on `build-data` (`--pmc --batch-size N`) extracts the shared MEDS in sequential N-encounter batches to `part-NNNN.parquet` files, bounding peak RAM to one batch — identical output, slower. Without it, a single `part-0000.parquet` is written.