"""flair-baseline CLI — staged subcommands, site-prefixed output, one shared MEDS.

Pipeline is split into stages you run in sequence (no per-task ETL re-runs):

  flair-baseline build-cohorts --clif-config clif.json --out .   # 5 cohort.parquet + table1
  flair-baseline build-data    --clif-config clif.json --out .   # ONE shared MEDS (ETL once)
  flair-baseline featurize     --clif-config clif.json --out .   # per-task features.npz + codes
  flair-baseline train         --clif-config clif.json --out .   # fit 5 models (fixed vocab)
  # — at a CLIF site, scope the data pull to the 25% holdout —
  flair-baseline build-data    … --holdout-only
  flair-baseline featurize     … --holdout-only
  flair-baseline infer  --models-dir mimic_baseline_models …
  # — maintainer, once — regenerate the committed feature vocabulary —
  flair-baseline build-vocab   --clif-config clif.json --out .

`prepare` chains build-cohorts → build-data → featurize for a single from-scratch run.

The CLIF→MEDS ETL runs ONCE over the union of all task cohorts (each join_id expanded
to its full stitched membership), written to `<site>_baseline_phi/_shared/MEDS/`. Every
task featurizes off that shared store. The feature vocabulary is a fixed, committed file
(`vocab.json`) shared by all tasks and all sites — train and infer both load it, so every
model has the same column space and bundles are interchangeable.

Artifacts partition by sensitivity (see layout.py):
  <site>_baseline_phi/_shared/MEDS/      shared events (PHI, stays local)
  <site>_baseline_phi/<task>/            cohort.parquet, features.npz, preds.parquet
  <site>_baseline_non_phi_for_upload/…   codes.parquet, table1.json, report/*.json (+ viz)
  <site>_baseline_models/<task>/         model.json, vocab.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import polars as pl
import typer

from flair_baseline.config import DEFAULT_ELF_CONFIG, resolve_task
from flair_baseline.featurize import (compute_counts, counts_to_X, load_counts,
                                      save_counts, truncated_code_map)
from flair_baseline.layout import SitePaths, TaskPaths, report_mode
from flair_baseline.train import train_and_score

DEFAULT_CLIF_CONFIG = "config/clif_config.template.json"
DEFAULT_VOCAB = "vocab.json"
JOIN_ID = "hospitalization_join_id"

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="XGBoost count-feature baseline for FLAIR tasks (staged pipeline)")


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #
def _auroc(preds: pl.DataFrame, label_col: str, split: str) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    d = preds.filter(pl.col("split") == split)
    if d.height == 0:
        return None
    y = d[label_col].to_numpy()
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, d["y_prob"].to_numpy()))


def _report_auroc(report_dir: Path) -> tuple:
    """Headline AUROC from discrimination.json, matching the report mode."""
    f = report_dir / "discrimination.json"
    if not f.exists():
        return None, ""
    d = json.loads(f.read_text())
    if isinstance(d.get("metrics"), dict):
        return d["metrics"].get("auroc"), ""
    lms = d.get("landmarks") or []
    if lms and isinstance(lms[0].get("metrics"), dict):
        return lms[0]["metrics"].get("auroc"), " lead-0"
    return None, ""


def _stitch(clif_config: str):
    """Build/load the encounter index once; return (clif cfg dict, index frame)."""
    from flair_benchmark._clif import read_clif_config
    from flair_benchmark._stitch import load_or_build_encounter_index
    cfg = read_clif_config(clif_config)
    idx = load_or_build_encounter_index(cfg, cfg.get("stitch_time_interval_hours", 6))
    n_blocks = idx[JOIN_ID].n_unique()
    typer.echo(f"[stitch] encounter index ready → {n_blocks:,} encounter blocks "
               f"from {idx.height:,} hospitalizations")
    return cfg, idx


def _resolve_tasks(task: Optional[str]) -> list[str]:
    from flair_benchmark.tasks import list_tasks
    return [resolve_task(task)] if task else list_tasks()


def _read_cohort(paths: TaskPaths, task_name: str, need: str) -> pl.DataFrame:
    """Load a task's cohort.parquet or exit with a 'run <need> first' message."""
    if not paths.cohort.exists():
        typer.echo(f"[{task_name}] no cohort.parquet at {paths.cohort} — run "
                   f"`flair-baseline {need}` first", err=True)
        raise typer.Exit(1)
    return pl.read_parquet(paths.cohort)


def _scope(cohort: pl.DataFrame, holdout_only: bool) -> pl.DataFrame:
    """Test-split rows for an inference site; the whole cohort for training."""
    return cohort.filter(pl.col("split") == "test") if holdout_only else cohort


def _load_vocab(vocab_path: str) -> list[str]:
    p = Path(vocab_path)
    if not p.exists():
        typer.echo(f"fixed vocabulary not found at {p} — run `flair-baseline build-vocab` "
                   f"(maintainer) or point --vocab at the committed vocab.json", err=True)
        raise typer.Exit(1)
    data = json.loads(p.read_text())
    return data["vocab"] if isinstance(data, dict) else list(data)


# --------------------------------------------------------------------------- #
# stage: build-cohorts
# --------------------------------------------------------------------------- #
def _do_build_cohorts(clif_config: str, out: str, tasks: list[str], site: str,
                      train_end: Optional[str], test_start: Optional[str]) -> None:
    from flair_benchmark._clif import read_clif_config
    from flair_benchmark._table1 import generate_table1
    from flair_benchmark.tasks import get_task

    for i, t in enumerate(tasks, 1):
        typer.echo(f"━━ build-cohorts {t} ({i}/{len(tasks)}) ━━")
        task_module = get_task(t)
        paths = TaskPaths.make(out, site, t)
        paths.mkdirs()
        # The flair contract: every cohort carries BOTH train+test splits, always.
        cohort = task_module.build(clif_config=clif_config,
                                   train_end=train_end, test_start=test_start)
        cohort.write_parquet(paths.cohort)
        n_tr = cohort.filter(pl.col("split") == "train").height
        n_te = cohort.filter(pl.col("split") == "test").height
        typer.echo(f"[{t}] cohort {cohort.height:,} rows ({n_tr:,} train / {n_te:,} test) "
                   f"→ {paths.cohort}")
        table1 = generate_table1(read_clif_config(clif_config), cohort, task_module)
        paths.table1.write_text(json.dumps(table1, indent=2, default=str))
        typer.echo(f"[{t}] Table 1 (total/train/test) → {paths.table1}")


# --------------------------------------------------------------------------- #
# stage: build-data (ONE shared MEDS over the union of all cohorts)
# --------------------------------------------------------------------------- #
def _manifest_key(join_ids: list[str], elf_cfg: dict, holdout_only: bool,
                  batch_size: Optional[int]) -> dict:
    jh = hashlib.sha1("\n".join(sorted(join_ids)).encode()).hexdigest()
    eh = hashlib.sha1(json.dumps(elf_cfg.get("domains", {}), sort_keys=True,
                                 default=str).encode()).hexdigest()
    return {"scope": "holdout" if holdout_only else "all", "n_joins": len(join_ids),
            "joins_hash": jh, "elf_hash": eh, "batch_size": batch_size}


def _do_build_data(clif_config: str, elf_config: str, out: str, tasks: list[str],
                   site: str, idx: pl.DataFrame, holdout_only: bool, reuse: bool,
                   batch_size: Optional[int]) -> None:
    from flair_benchmark._clif import read_clif_config
    from flair_benchmark._stitch import members_of_joins
    from flair_benchmark.features.fe_meds import build_shared_meds, load_elf_config

    sp = SitePaths.make(out, site)
    sp.mkdirs()

    # Union every task cohort's join ids (test-split only at an inference site).
    join_ids: set[str] = set()
    for t in tasks:
        cohort = _scope(_read_cohort(TaskPaths.make(out, site, t), t, "build-cohorts"),
                        holdout_only)
        join_ids |= set(cohort[JOIN_ID].cast(pl.Utf8).to_list())
    join_ids_l = sorted(join_ids)
    elf_cfg = load_elf_config(elf_config)
    key = _manifest_key(join_ids_l, elf_cfg, holdout_only, batch_size)

    if reuse and sp.has_meds() and sp.manifest.exists():
        if json.loads(sp.manifest.read_text()).get("key") == key:
            typer.echo(f"[build-data] reusing shared MEDS ({key['scope']}, "
                       f"{key['n_joins']:,} encounters) — manifest match")
            return

    # Expand each encounter to ALL member hospitalizations (siblings included), so
    # no stitched event is dropped just because its member isn't a prediction row.
    hosp_ids = members_of_joins(idx, join_ids_l)
    mode = (f"-pmc batched (size {batch_size})" if batch_size else "single pass")
    typer.echo(f"[build-data] {key['scope']}: {len(join_ids_l):,} encounters → "
               f"{len(hosp_ids):,} member hospitalizations — ETL {mode}")
    versions = build_shared_meds(read_clif_config(clif_config), elf_cfg, hosp_ids, idx,
                                 str(sp.shared_root), batch_size=batch_size)
    sp.versions.write_text(json.dumps(versions, indent=2))
    sp.manifest.write_text(json.dumps({"key": key, "versions": versions}, indent=2))
    typer.echo(f"[build-data] shared MEDS → {sp.meds_dir}  (manifest → {sp.manifest})")


# --------------------------------------------------------------------------- #
# stage: featurize (per task, off the shared MEDS)
# --------------------------------------------------------------------------- #
def _do_featurize(clif_config: str, out: str, tasks: list[str], site: str,
                  holdout_only: bool) -> None:
    from flair_benchmark.meds_etl import build_codes_table

    sp = SitePaths.make(out, site)
    if not sp.has_meds():
        typer.echo(f"[featurize] no shared MEDS at {sp.meds_dir} — run "
                   f"`flair-baseline build-data` first", err=True)
        raise typer.Exit(1)
    versions = json.loads(sp.versions.read_text()) if sp.versions.exists() else {}

    for i, t in enumerate(tasks, 1):
        typer.echo(f"━━ featurize {t} ({i}/{len(tasks)}) ━━")
        paths = TaskPaths.make(out, site, t)
        paths.mkdirs()
        score_cohort = _scope(_read_cohort(paths, t, "build-cohorts"), holdout_only)

        # Restrict the shared store to this task's encounters up front: keeps the
        # code-space scan + count join scoped, and the inner join in compute_counts
        # would drop the rest anyway. Sibling events (same join_id) are included.
        joins = score_cohort[JOIN_ID].cast(pl.Utf8).unique().to_list()
        ev_task = (pl.scan_parquet(sp.meds_glob)
                   .with_columns(pl.col(JOIN_ID).cast(pl.Utf8))
                   .filter(pl.col(JOIN_ID).is_in(joins)))

        counts = compute_counts(ev_task, score_cohort, desc=f"[{t}] featurize")
        save_counts(paths.features, counts)
        typer.echo(f"[{t}] features → {paths.features} "
                   f"({counts.n_rows:,} rows, {len(counts.trunc_by_cint):,} codes)")

        # Per-task codes registry (non-PHI upload), built from this task's events.
        codes_events = ev_task.select("code", "numeric_value", "text_value").collect(
            engine="streaming")
        codes = build_codes_table(codes_events, versions)
        codes.write_parquet(paths.codes)
        typer.echo(f"[{t}] codes → {paths.codes} ({codes.height:,} codes)")


# --------------------------------------------------------------------------- #
# stage: report (shared by train + infer)
# --------------------------------------------------------------------------- #
def _report(paths: TaskPaths, task_name: str, clif_config: str, preds: pl.DataFrame,
            label_col: str, report: bool, viz: bool) -> None:
    from flair_benchmark._clif import read_clif_config

    auc_tr, auc_te = _auroc(preds, label_col, "train"), _auroc(preds, label_col, "test")
    if not report:
        typer.echo(f"[{task_name}] row-level AUROC  train={auc_tr}  test={auc_te}")
        return
    from flair_benchmark.report import build_report
    from flair_benchmark.tasks import get_task
    site = read_clif_config(clif_config).get("site")
    mode = report_mode(task_name)
    build_report(str(paths.preds), get_task(task_name), str(paths.report_dir),
                 cohort_path=str(paths.cohort), viz=viz, site=site, mode=mode)
    rep_auc, lbl = _report_auroc(paths.report_dir)
    typer.echo(f"[{task_name}] report AUROC ({mode}{lbl}, test)={rep_auc}  "
               f"[row-level train={auc_tr} test={auc_te}]")
    typer.echo(f"[{task_name}] report ({mode}){' + viz' if viz else ''} → {paths.report_dir}")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
@app.command("build-cohorts")
def build_cohorts_cmd(
    clif_config: str = typer.Option(DEFAULT_CLIF_CONFIG, "--clif-config"),
    out: str = typer.Option(".", "--out"),
    task: Optional[str] = typer.Option(None, "--task", help="Task name/prefix; default = all 5"),
    train_end: Optional[str] = typer.Option(None, "--train-end"),
    test_start: Optional[str] = typer.Option(None, "--test-start"),
) -> None:
    """Build each task's cohort.parquet (always train+test) + table1.json."""
    cfg, _ = _stitch(clif_config)
    _do_build_cohorts(clif_config, out, _resolve_tasks(task), cfg.get("site"),
                      train_end, test_start)


@app.command("build-data")
def build_data_cmd(
    clif_config: str = typer.Option(DEFAULT_CLIF_CONFIG, "--clif-config"),
    elf_config: str = typer.Option(DEFAULT_ELF_CONFIG, "--elf-config"),
    out: str = typer.Option(".", "--out"),
    task: Optional[str] = typer.Option(None, "--task", help="Default = union of all 5"),
    holdout_only: bool = typer.Option(False, "--holdout-only/--full-cohort",
                                      help="Pull ETL for the 25%% test split only (inference sites)"),
    reuse: bool = typer.Option(False, "--reuse/--no-reuse",
                               help="Skip rebuild if the shared MEDS manifest matches"),
    pmc: bool = typer.Option(False, "--pmc/--no-pmc", hidden=True,
                             help="poor-man's-compute: batch the ETL for bounded RAM"),
    batch_size: int = typer.Option(4000, "--batch-size", hidden=True,
                                   help="encounters per ETL batch when --pmc"),
) -> None:
    """Build the ONE shared MEDS store (CLIF→MEDS ETL, run once) over all cohorts."""
    cfg, idx = _stitch(clif_config)
    _do_build_data(clif_config, elf_config, out, _resolve_tasks(task), cfg.get("site"),
                   idx, holdout_only, reuse, batch_size if pmc else None)


@app.command("featurize")
def featurize_cmd(
    clif_config: str = typer.Option(DEFAULT_CLIF_CONFIG, "--clif-config"),
    out: str = typer.Option(".", "--out"),
    task: Optional[str] = typer.Option(None, "--task", help="Task name/prefix; default = all 5"),
    holdout_only: bool = typer.Option(False, "--holdout-only/--full-cohort",
                                      help="Featurize the 25%% test split only (inference sites)"),
) -> None:
    """Count-featurize each task off the shared MEDS → features.npz + codes.parquet."""
    from flair_benchmark._clif import read_clif_config
    site = read_clif_config(clif_config).get("site")
    _do_featurize(clif_config, out, _resolve_tasks(task), site, holdout_only)


@app.command("train")
def train_cmd(
    clif_config: str = typer.Option(DEFAULT_CLIF_CONFIG, "--clif-config"),
    out: str = typer.Option(".", "--out"),
    task: Optional[str] = typer.Option(None, "--task", help="Task name/prefix; default = all 5"),
    vocab: str = typer.Option(DEFAULT_VOCAB, "--vocab",
                              help="Committed fixed feature vocabulary (vocab.json)"),
    report: bool = typer.Option(True, "--report/--no-report"),
    viz: bool = typer.Option(True, "--viz/--no-viz"),
    hpo: bool = typer.Option(True, "--hpo/--no-hpo"),
    hpo_trials: int = typer.Option(30, "--hpo-trials"),
) -> None:
    """Fit XGBoost per task on prepared features (fixed vocab); models land in <site>_baseline_models/."""
    from flair_benchmark._clif import read_clif_config
    from flair_benchmark.tasks import get_task

    site = read_clif_config(clif_config).get("site")
    fixed_vocab = _load_vocab(vocab)
    n_trials = hpo_trials if hpo else 0
    tasks = _resolve_tasks(task)
    for i, t in enumerate(tasks, 1):
        typer.echo(f"━━ train {t} ({i}/{len(tasks)}) ━━")
        label_col = get_task(t).META["label_column"]
        paths = TaskPaths.make(out, site, t)
        paths.mkdirs()
        cohort = _read_cohort(paths, t, "build-cohorts")
        counts = load_counts(paths.features, cohort)
        if counts is None:
            typer.echo(f"[{t}] no matching features.npz — run `flair-baseline featurize` "
                       f"first", err=True)
            raise typer.Exit(1)
        X, ids, _ = counts_to_X(counts, vocab=fixed_vocab)
        typer.echo(f"[{t}] feature matrix: {X.shape[0]:,} × {X.shape[1]:,} ({X.nnz:,} nnz) "
                   f"— training XGBoost ({f'HPO {n_trials} trials' if n_trials else 'fixed params'})")
        preds = train_and_score(X, ids, fixed_vocab, cohort, label_col,
                                model_out=str(paths.model), vocab_out=str(paths.vocab),
                                params_out=str(paths.params), n_trials=n_trials)
        preds.write_parquet(paths.preds)
        _report(paths, t, clif_config, preds, label_col, report, viz)


@app.command("infer")
def infer_cmd(
    models_dir: str = typer.Option(..., "--models-dir"),
    clif_config: str = typer.Option(DEFAULT_CLIF_CONFIG, "--clif-config"),
    out: str = typer.Option(".", "--out"),
    task: Optional[str] = typer.Option(None, "--task", help="Task name/prefix; default = all 5"),
    report: bool = typer.Option(True, "--report/--no-report"),
    viz: bool = typer.Option(True, "--viz/--no-viz"),
) -> None:
    """Score shipped models on this site's prepared (holdout) features; report on the 25% test split."""
    from flair_benchmark._clif import read_clif_config
    from flair_benchmark.tasks import get_task

    site = read_clif_config(clif_config).get("site")
    models_root = Path(models_dir)
    tasks = _resolve_tasks(task)
    for i, t in enumerate(tasks, 1):
        typer.echo(f"━━ infer {t} ({i}/{len(tasks)}) ━━")
        model_path = models_root / t / "model.json"
        vocab_path = models_root / t / "vocab.json"
        if not (model_path.exists() and vocab_path.exists()):
            typer.echo(f"[{t}] no model under {models_root / t} — skipping")
            continue
        vocab_list = json.loads(vocab_path.read_text())["vocab"]
        label_col = get_task(t).META["label_column"]
        paths = TaskPaths.make(out, site, t)
        paths.mkdirs()
        score_cohort = _read_cohort(paths, t, "build-cohorts").filter(pl.col("split") == "test")
        counts = load_counts(paths.features, score_cohort)
        if counts is None:
            typer.echo(f"[{t}] no matching features.npz — run `flair-baseline featurize "
                       f"--holdout-only` first", err=True)
            raise typer.Exit(1)
        X, ids, _ = counts_to_X(counts, vocab=vocab_list)
        typer.echo(f"[{t}] feature matrix: {X.shape[0]:,} × {X.shape[1]:,} ({X.nnz:,} nnz) "
                   f"— scoring with shipped model")
        preds = train_and_score(X, ids, vocab_list, score_cohort, label_col,
                                model_in=str(model_path))
        preds.write_parquet(paths.preds)
        _report(paths, t, clif_config, preds, label_col, report, viz)


@app.command("build-vocab")
def build_vocab_cmd(
    clif_config: str = typer.Option(DEFAULT_CLIF_CONFIG, "--clif-config"),
    out: str = typer.Option(".", "--out"),
    vocab_out: str = typer.Option(DEFAULT_VOCAB, "--vocab-out",
                                  help="Where to write the committed fixed vocabulary"),
) -> None:
    """Regenerate the fixed feature vocabulary from the shared MEDS (maintainer, once)."""
    from flair_benchmark._clif import read_clif_config

    site = read_clif_config(clif_config).get("site")
    sp = SitePaths.make(out, site)
    if not sp.has_meds():
        typer.echo(f"no shared MEDS at {sp.meds_dir} — run `flair-baseline build-data` first",
                   err=True)
        raise typer.Exit(1)
    code_map = truncated_code_map(pl.scan_parquet(sp.meds_glob))
    vocab = code_map.select("trunc").unique().sort("trunc")["trunc"].to_list()
    Path(vocab_out).write_text(json.dumps({"vocab": vocab}, indent=2))
    typer.echo(f"[build-vocab] {len(vocab):,} codes → {vocab_out}")


@app.command("prepare")
def prepare_cmd(
    clif_config: str = typer.Option(DEFAULT_CLIF_CONFIG, "--clif-config"),
    elf_config: str = typer.Option(DEFAULT_ELF_CONFIG, "--elf-config"),
    out: str = typer.Option(".", "--out"),
    task: Optional[str] = typer.Option(None, "--task"),
    train_end: Optional[str] = typer.Option(None, "--train-end"),
    test_start: Optional[str] = typer.Option(None, "--test-start"),
    holdout_only: bool = typer.Option(False, "--holdout-only/--full-cohort"),
    reuse: bool = typer.Option(False, "--reuse/--no-reuse"),
    pmc: bool = typer.Option(False, "--pmc/--no-pmc", hidden=True),
    batch_size: int = typer.Option(4000, "--batch-size", hidden=True),
) -> None:
    """Convenience: build-cohorts → build-data → featurize in one go."""
    cfg, idx = _stitch(clif_config)
    site = cfg.get("site")
    tasks = _resolve_tasks(task)
    _do_build_cohorts(clif_config, out, tasks, site, train_end, test_start)
    _do_build_data(clif_config, elf_config, out, tasks, site, idx, holdout_only, reuse,
                   batch_size if pmc else None)
    _do_featurize(clif_config, out, tasks, site, holdout_only)
    typer.echo("[prepare] done — run `flair-baseline train` or `infer` next")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
