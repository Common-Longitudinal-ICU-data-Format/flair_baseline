"""Guards on the flair-benchmark surface this baseline consumes.

The baseline vendors flair-benchmark as a prebuilt wheel under ``wheels/``, so a
benchmark change lands here only when someone rebuilds it. These tests are the
tripwire for that moment: they fail in seconds on the next incompatible rebuild
instead of hours into a MIMIC run.

Both cases are regressions we actually hit when the benchmark was reorganized
into pipeline-stage packages:

  * ``report_mode`` used to come from a ``TASK_POLICY`` dict that was deleted, and
    returned modes (``landmark``/``peak``) that ``build_report`` now rejects
    outright with a ValueError.
  * the report bundle consolidated (schema 3 -> 4) and ``discrimination.json``
    stopped existing, so the headline-AUROC reader silently returned None.
"""
from __future__ import annotations

import json

import pytest
from flair_baseline.cli import _headline_json, _report_auroc
from flair_baseline.layout import report_mode

from flair_baseline.config import available_tasks
from flair_benchmark.constants import REPORT_MODES


# --------------------------------------------------------------------------- #
# report_mode must stay inside the benchmark's accepted vocabulary
# --------------------------------------------------------------------------- #
def test_every_task_reports_a_mode_the_benchmark_accepts():
    """``report_mode`` feeds straight into ``build_report(mode=...)``.

    ``report._resolve_mode`` raises on anything outside REPORT_MODES, so a task
    whose META drifts out of that set would crash at report time — after the full
    ETL, featurization and model fit have already been paid for.
    """
    tasks = available_tasks()
    assert tasks, "no tasks discovered — the benchmark task registry is empty"
    for t in tasks:
        assert report_mode(t) in REPORT_MODES, (
            f"task {t!r} reports mode {report_mode(t)!r}, which build_report "
            f"rejects; accepted: {sorted(REPORT_MODES)}")


def test_report_mode_falls_back_for_an_unknown_task():
    """An unresolvable task name must not raise — layout owns the fallback."""
    assert report_mode("no_such_task_exists") in REPORT_MODES


# --------------------------------------------------------------------------- #
# _report_auroc must read the schema-4 bundle, and tolerate absent metrics
# --------------------------------------------------------------------------- #
def test_report_auroc_reads_episodic_overall(tmp_path):
    (tmp_path / "overall.json").write_text(json.dumps(
        {"discrimination": {"auroc": 0.812, "auprc": 0.4}}))
    assert _report_auroc(tmp_path, "episodic") == (0.812, "")


def test_report_auroc_reads_continuous_pooled(tmp_path):
    """Continuous tasks report the pooled aggregate, NOT the first landmark.

    Guards the intended change of headline number: schema 3 reported lead-0, so a
    reader that silently fell back to by_landmark[0] would look fine while
    reporting a different quantity.
    """
    (tmp_path / "landmark.json").write_text(json.dumps({
        "pooled": {"auroc": 0.741},
        "by_landmark": [{"hospitalization_time": 1, "auroc": 0.999}],
    }))
    assert _report_auroc(tmp_path, "continuous") == (0.741, " pooled")


@pytest.mark.parametrize("mode", ["episodic", "continuous"])
def test_report_auroc_tolerates_missing_bundle(tmp_path, mode):
    """An empty report dir yields None, not an exception."""
    auroc, _ = _report_auroc(tmp_path, mode)
    assert auroc is None


@pytest.mark.parametrize(
    "mode,filename,payload",
    [
        # A degenerate split (single label class) legitimately nulls the metric.
        ("episodic", "overall.json", {"discrimination": {"auroc": None}}),
        ("continuous", "landmark.json", {"pooled": {"auroc": None}}),
        # A suppressed/degenerate report may omit the section entirely.
        ("episodic", "overall.json", {"metadata": {"reason": "single label class"}}),
        ("continuous", "landmark.json", {"metadata": {"reason": "no landmarks"}}),
    ],
)
def test_report_auroc_tolerates_null_metrics(tmp_path, mode, filename, payload):
    (tmp_path / filename).write_text(json.dumps(payload))
    auroc, _ = _report_auroc(tmp_path, mode)
    assert auroc is None


def test_report_auroc_does_not_read_the_deleted_discrimination_json(tmp_path):
    """discrimination.json is schema 3. Reading it again would resurrect the bug."""
    (tmp_path / "discrimination.json").write_text(json.dumps(
        {"metrics": {"auroc": 0.99}}))
    for mode in ("episodic", "continuous"):
        auroc, _ = _report_auroc(tmp_path, mode)
        assert auroc is None


# --------------------------------------------------------------------------- #
# a plotting failure must not discard a finished run
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode,filename", [("episodic", "overall.json"),
                                           ("continuous", "landmark.json")])
def test_headline_json_matches_the_auroc_reader(tmp_path, mode, filename):
    """The viz-tolerance check and the AUROC reader must agree on the filename.

    If they drift, a viz failure either aborts a good run or masks a genuinely
    missing report.
    """
    assert _headline_json(tmp_path, mode).name == filename
    (tmp_path / filename).write_text(json.dumps(
        {"discrimination": {"auroc": 0.5}, "pooled": {"auroc": 0.5}}))
    assert _headline_json(tmp_path, mode).exists()
    assert _report_auroc(tmp_path, mode)[0] == 0.5


def test_sweep_band_filter_rejects_half_open_ci():
    """flair_benchmark's threshold-sweep band must skip half-open CIs.

    NNE is 1/PPV, so a PPV bound at zero yields nne_ci = [4.8, None]. A plain
    truthiness test lets that reach matplotlib's fill_between, whose isfinite
    raises TypeError — which killed every episodic task once matplotlib was
    actually installed. Assert the numeric-endpoint guard is present upstream.
    """
    import inspect

    from flair_benchmark.report import _viz

    src = inspect.getsource(_viz._sweep_fig)
    assert "isinstance" in src, (
        "flair_benchmark.report._viz._sweep_fig no longer guards CI endpoints for "
        "numeric type; a half-open CI like [4.8, None] will crash fill_between")
