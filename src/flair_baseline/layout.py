"""Site-prefixed, sensitivity-partitioned output layout.

Every run writes into three sibling folders under one ``--out`` root, each
prefixed with the site name (from the clif config ``site`` field):

  <site>_baseline_phi/<task>/                 cohort.parquet, MEDS.parquet, preds_<kind>.parquet
  <site>_baseline_non_phi_for_upload/<task>/  codes.parquet, table1.json, report/<kind>/…
  <site>_baseline_models/<task>/<kind>/       model.json, vocab.json, params.json

Only the non-PHI folder is meant to leave the site. Models are what
`local-training` ships to other sites; `external-validation` and
`transfer-learning` read them back in and regenerate the PHI + non-PHI folders
locally against the new site's data.

A site can hold three answers at once, so every *per-model* artifact carries a
``kind`` segment (see ``KINDS``). Artifacts that describe the site's **data**
rather than a model — cohort, features, codes, table1 — have no kind and are
never duplicated.

`TaskPaths` is the single owner of the path mapping so every command routes
writes identically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from flair_benchmark.tasks import get_task

# --------------------------------------------------------------------------- #
# model kinds — the one axis that separates a site's three answers
# --------------------------------------------------------------------------- #
#   external_validation  a shipped model, frozen, scored on this site's holdout
#   transfer             that shipped model, boosted further on local train rows
#   local                a model fit on local train rows only
# The source site (MIMIC) produces `local` and ships it; every other site can
# produce all three and compare them on the same 25% holdout.
KIND_EXTERNAL = "external_validation"
KIND_TRANSFER = "transfer"
KIND_LOCAL = "local"
KINDS = (KIND_EXTERNAL, KIND_TRANSFER, KIND_LOCAL)


def check_kind(kind: str) -> str:
    """Validate a kind at the path boundary, so a typo can't invent a folder."""
    if kind not in KINDS:
        raise ValueError(f"unknown model kind {kind!r}; expected one of {', '.join(KINDS)}")
    return kind


def report_mode(task_name: str) -> str:
    """Report mode for a task — read from the task module's own ``META``.

    Continuous tasks report per lead-time; episode tasks report episodic.
    An unknown task (``get_task`` raises) defaults to episodic. The lookup stays
    generic, so a task whose META names another mode still routes correctly.
    """
    try:
        return get_task(task_name).META.get("report_mode", "episodic")
    except (ValueError, TypeError):
        return "episodic"


def slug(site: str | None) -> str:
    """Folder-safe site token. Raises when empty so folders are never un-prefixed."""
    s = re.sub(r"[^a-z0-9]+", "_", (site or "").strip().lower()).strip("_")
    if not s:
        raise ValueError(
            "clif config has no usable 'site' name; set \"site\" in the config "
            "(it prefixes every output folder).")
    return s


@dataclass(frozen=True)
class SitePaths:
    """Site-level (task-independent) artifacts: the ONE shared MEDS store.

    The CLIF→MEDS ETL is built once over the union of all task cohorts and lives
    here, under the PHI root, so every task featurizes off the same events instead
    of re-extracting. ``MEDS/`` holds one or more ``part-NNNN.parquet`` files
    (a single part for a normal run; several for the batched ``-pmc`` mode), all
    scanned together via the ``meds_glob``.
    """

    out_root: Path
    site: str

    @classmethod
    def make(cls, out_root: str | Path, site: str) -> "SitePaths":
        return cls(Path(out_root), slug(site))

    @property
    def shared_root(self) -> Path:
        return self.out_root / f"{self.site}_baseline_phi" / "_shared"

    @property
    def meds_dir(self) -> Path:
        return self.shared_root / "MEDS"

    @property
    def meds_glob(self) -> str:
        """Glob passed to ``pl.scan_parquet`` — covers single- and multi-part stores."""
        return str(self.meds_dir / "part-*.parquet")

    @property
    def versions(self) -> Path:
        """Per-domain ELF versions captured at build-data time (for the codes registry)."""
        return self.shared_root / "domain_versions.json"

    @property
    def manifest(self) -> Path:
        """Reuse key for build-data: scope + join-id-set hash + elf hash + batch info."""
        return self.shared_root / "manifest.json"

    def has_meds(self) -> bool:
        return self.meds_dir.exists() and any(self.meds_dir.glob("part-*.parquet"))

    def mkdirs(self) -> None:
        self.meds_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TaskPaths:
    """Absolute paths for one task's artifacts across the three folders."""

    out_root: Path
    site: str
    task: str

    @classmethod
    def make(cls, out_root: str | Path, site: str, task: str) -> "TaskPaths":
        return cls(Path(out_root), slug(site), task)

    # --- roots (site-prefixed) ---------------------------------------------
    @property
    def phi_root(self) -> Path:
        return self.out_root / f"{self.site}_baseline_phi" / self.task

    @property
    def nonphi_root(self) -> Path:
        return self.out_root / f"{self.site}_baseline_non_phi_for_upload" / self.task

    @property
    def models_root(self) -> Path:
        return self.out_root / f"{self.site}_baseline_models" / self.task

    # --- PHI (stays local) -------------------------------------------------
    @property
    def cohort(self) -> Path:
        return self.phi_root / "cohort.parquet"

    @property
    def meds(self) -> Path:
        return self.phi_root / "data" / "MEDS.parquet"

    def preds(self, kind: str) -> Path:
        """Per-kind predictions. Without the suffix the second pipeline to run at
        a site would overwrite the first one's preds, and the report built from
        them would be attributed to the wrong model."""
        return self.phi_root / f"preds_{check_kind(kind)}.parquet"

    @property
    def features(self) -> Path:
        """Cached feature-join output (sidecar features_meta.json sits beside it)."""
        return self.phi_root / "features.npz"

    # --- non-PHI (uploaded) ------------------------------------------------
    @property
    def codes(self) -> Path:
        return self.nonphi_root / "codes.parquet"

    @property
    def table1(self) -> Path:
        return self.nonphi_root / "table1.json"

    def report_dir(self, kind: str) -> Path:
        return self.nonphi_root / "report" / check_kind(kind)

    # --- models (shipped to other sites) -----------------------------------
    def model_dir(self, kind: str) -> Path:
        return self.models_root / check_kind(kind)

    def model(self, kind: str) -> Path:
        return self.model_dir(kind) / "model.json"

    def vocab(self, kind: str) -> Path:
        return self.model_dir(kind) / "vocab.json"

    def params(self, kind: str) -> Path:
        """Best XGBoost hyperparameters from HPO (informational + shippable)."""
        return self.model_dir(kind) / "params.json"

    def mkdirs(self, kind: str | None = None) -> None:
        """Create every kind-independent directory this task writes into, plus
        the per-kind report/model dirs when a kind is given."""
        dirs = [self.meds.parent, self.phi_root, self.codes.parent]
        if kind is not None:
            dirs += [self.report_dir(kind), self.model_dir(kind)]
        for p in dirs:
            p.mkdir(parents=True, exist_ok=True)


def resolve_shipped_model(models_root: Path, task: str) -> tuple[Path, Path] | None:
    """Locate a shipped (model.json, vocab.json) pair for ``task``.

    Bundles built after the kind split live at ``<task>/local/``; bundles built
    before it are flat at ``<task>/``. Try the current layout first, fall back to
    the flat one, and return None when neither is complete.
    """
    for d in (models_root / task / KIND_LOCAL, models_root / task):
        model, vocab = d / "model.json", d / "vocab.json"
        if model.exists() and vocab.exists():
            return model, vocab
    return None
