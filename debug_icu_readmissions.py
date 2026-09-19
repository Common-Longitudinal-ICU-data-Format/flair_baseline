"""Audit ICU readmissions from raw ADT after encounter stitching.

Edit the paths below, then run:

    uv run python debug_icu_readmissions.py

The two output files contain PHI and are deliberately written beside the input
cohort under ``*_baseline_phi``.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from flair_benchmark.cohort.events import (
    ALLOWED_ICU_GAP_LOCATIONS,
    LOCATION_MAPPING,
    _merge_runs,
)
from flair_benchmark.cohort.stitch import load_or_build_encounter_index, members_of_joins


# ---------------------------------------------------------------------------
# Settings: these are the only lines most sites need to change.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SITE_NAME = "rush"
CLIF_CONFIG_PATH = ROOT / "config/clif_config_rush.json"
COHORT_PATH = ROOT / "rush_baseline_phi/icu_readmission/cohort.parquet"
SUMMARY_OUTPUT_PATH = COHORT_PATH.with_name("readmission_debug.parquet")
EVENT_OUTPUT_PATH = COHORT_PATH.with_name("readmission_events_debug.parquet")

JOIN_ID = "hospitalization_join_id"
RAW_HOSP_ID = "source_hospitalization_id"
LABEL = "label_icu_readmission"


def _read_config() -> dict:
    cfg = json.loads(CLIF_CONFIG_PATH.read_text())
    if cfg.get("site") != SITE_NAME:
        raise ValueError(
            f"SITE_NAME is {SITE_NAME!r}, but {CLIF_CONFIG_PATH} says "
            f"{cfg.get('site')!r}"
        )
    cache_dir = Path(cfg.get("cache_directory", f"output_{SITE_NAME}"))
    if not cache_dir.is_absolute():
        cfg["cache_directory"] = str(ROOT / cache_dir)
    return cfg


def _read_adt(cfg: dict, hospitalization_ids: list[str]) -> pl.DataFrame:
    filetype = cfg["filetype"].lower()
    path = Path(cfg["data_directory"]) / f"clif_adt.{filetype}"
    if not path.exists():
        raise FileNotFoundError(f"ADT file not found: {path}")

    columns = [
        "hospitalization_id",
        "location_category",
        "in_dttm",
        "out_dttm",
    ]
    if filetype == "parquet":
        frame = pl.scan_parquet(path).select(columns)
    elif filetype == "csv":
        frame = pl.scan_csv(path, try_parse_dates=True).select(columns)
    else:
        raise ValueError(f"Unsupported filetype {filetype!r}; use parquet or csv")

    return (
        frame.with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
        .filter(pl.col("hospitalization_id").is_in(hospitalization_ids))
        .collect()
    )


def _standardize_locations(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("location_category")
        .str.to_lowercase()
        .alias("raw_location"),
        pl.col("location_category")
        .str.to_lowercase()
        .replace_strict(LOCATION_MAPPING, default="Other")
        .alias("std_location"),
    )


def _normalized_runs(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return location runs before and after production's allowed-gap removal."""
    first_pass = _merge_runs(df)
    with_neighbors = first_pass.with_columns(
        pl.col("std_location").shift(1).over("hospitalization_id").alias("_prev"),
        pl.col("std_location").shift(-1).over("hospitalization_id").alias("_next"),
    )
    without_allowed_gaps = with_neighbors.filter(
        ~(
            pl.col("std_location").is_in(ALLOWED_ICU_GAP_LOCATIONS)
            & (pl.col("_prev") == "ICU")
            & (pl.col("_next") == "ICU")
        )
    ).drop("_prev", "_next")
    return first_pass, _merge_runs(without_allowed_gaps)


def _run_sequence(runs: pl.DataFrame) -> pl.DataFrame:
    return (
        runs.rename({"hospitalization_id": JOIN_ID})
        .group_by(JOIN_ID, maintain_order=True)
        .agg(pl.col("std_location").alias("_locations"))
        .with_columns(
            pl.col("_locations").list.join(" -> ").alias("location_sequence")
        )
        .drop("_locations")
    )


def _icu_events(runs: pl.DataFrame) -> pl.DataFrame:
    runs = runs.rename({"hospitalization_id": JOIN_ID}).sort([JOIN_ID, "in_dttm"])
    tagged = runs.with_columns(
        (pl.col("std_location") == "ICU")
        .cast(pl.Int32)
        .cum_sum()
        .over(JOIN_ID)
        .alias("_previous_icu_number")
    )
    between = (
        tagged.filter(
            (pl.col("std_location") != "ICU")
            & (pl.col("_previous_icu_number") > 0)
        )
        .group_by([JOIN_ID, "_previous_icu_number"], maintain_order=True)
        .agg(pl.col("std_location").alias("between_icu_locations"))
    )

    icu = runs.filter(pl.col("std_location") == "ICU").with_columns(
        pl.col("in_dttm").rank("ordinal").over(JOIN_ID).cast(pl.Int32).alias(
            "icu_stay_number"
        ),
        pl.col("in_dttm").shift(1).over(JOIN_ID).alias("previous_icu_start_dttm"),
        pl.col("out_dttm").shift(1).over(JOIN_ID).alias("previous_icu_end_dttm"),
    )
    returns = (
        icu.filter(pl.col("icu_stay_number") > 1)
        .rename({"in_dttm": "return_icu_start_dttm", "out_dttm": "return_icu_end_dttm"})
        .with_columns(
            (pl.col("icu_stay_number") - 1).alias("_previous_icu_number")
        )
        .join(between, on=[JOIN_ID, "_previous_icu_number"], how="left")
        .with_columns(
            (
                (pl.col("return_icu_start_dttm") - pl.col("previous_icu_end_dttm"))
                .dt.total_seconds()
                / 3600.0
            ).alias("hours_between_icu_stays"),
            pl.concat_str(
                [
                    pl.lit("ICU"),
                    pl.col("between_icu_locations").list.join(" -> "),
                    pl.lit("ICU"),
                ],
                separator=" -> ",
            ).alias("return_path"),
        )
    )

    locations = pl.col("between_icu_locations")
    single = locations.list.len() == 1
    has_ward = locations.list.contains("Ward")
    has_other = locations.list.contains("Other")
    has_er = locations.list.contains("ER")
    has_allowed = (
        locations.list.contains("Procedural")
        | locations.list.contains("Radiology")
        | locations.list.contains("Dialysis")
    )
    return returns.with_columns(
        pl.when(has_er)
        .then(pl.lit("er"))
        .when(has_ward & has_other)
        .then(pl.lit("ward_other_mixed"))
        .when(has_ward)
        .then(pl.lit("ward"))
        .when(has_other)
        .then(pl.lit("other"))
        .when(has_allowed)
        .then(pl.lit("allowed_location_only"))
        .otherwise(pl.lit("other_complex"))
        .alias("return_route_family"),
        pl.when(single & (locations.list.first() == "Ward"))
        .then(pl.lit("icu_ward_icu"))
        .when(single & (locations.list.first() == "Other"))
        .then(pl.lit("icu_other_icu"))
        .when(has_er)
        .then(pl.lit("icu_er_icu"))
        .when(has_ward & has_other & has_allowed)
        .then(pl.lit("icu_ward_other_with_allowed_location_icu"))
        .when(has_ward & has_other)
        .then(pl.lit("icu_ward_other_mixed_icu"))
        .when(has_ward & has_allowed)
        .then(pl.lit("icu_ward_with_allowed_location_icu"))
        .when(has_other & has_allowed)
        .then(pl.lit("icu_other_with_allowed_location_icu"))
        .when(has_ward)
        .then(pl.lit("icu_ward_complex_icu"))
        .when(has_other)
        .then(pl.lit("icu_other_complex_icu"))
        .when(has_allowed)
        .then(pl.lit("icu_allowed_location_only_icu"))
        .otherwise(pl.lit("icu_unclassified_complex_icu"))
        .alias("return_path_category"),
    )


def _raw_quality(adt: pl.DataFrame) -> pl.DataFrame:
    ordered = (
        adt.sort([JOIN_ID, "in_dttm", "out_dttm"])
        .with_columns(pl.col("out_dttm").cum_max().over(JOIN_ID).alias("_max_out"))
        .with_columns(
            pl.col("_max_out").shift(1).over(JOIN_ID).alias("_previous_max_out"),
            pl.struct(
                [RAW_HOSP_ID, "in_dttm", "out_dttm", "raw_location"]
            ).is_duplicated().alias("_duplicate"),
        )
        .with_columns(
            (pl.col("in_dttm") < pl.col("_previous_max_out"))
            .fill_null(False)
            .alias("_overlap")
        )
    )
    simultaneous = (
        ordered.group_by([JOIN_ID, "in_dttm"])
        .agg(pl.col("std_location").n_unique().alias("_location_count"))
        .group_by(JOIN_ID)
        .agg((pl.col("_location_count") > 1).any().alias("has_simultaneous_locations"))
    )
    quality = ordered.group_by(JOIN_ID).agg(
        (pl.col("out_dttm") <= pl.col("in_dttm"))
        .any()
        .alias("has_nonpositive_duration"),
        pl.col("_overlap").any().alias("has_overlapping_intervals"),
        pl.col("_duplicate").any().alias("has_duplicate_intervals"),
        pl.col("raw_location")
        .filter(~pl.col("raw_location").is_in(list(LOCATION_MAPPING)))
        .unique()
        .sort()
        .alias("unmapped_locations"),
    )
    return quality.join(simultaneous, on=JOIN_ID, how="left")


def _add_event_provenance(events: pl.DataFrame, adt: pl.DataFrame) -> pl.DataFrame:
    icu_owners = (
        adt.filter(pl.col("std_location") == "ICU")
        .sort([JOIN_ID, "in_dttm", RAW_HOSP_ID])
        .group_by([JOIN_ID, "in_dttm"])
        .agg(pl.col(RAW_HOSP_ID).first().alias("_owner"))
    )
    previous_owner = icu_owners.rename(
        {
            "in_dttm": "previous_icu_start_dttm",
            "_owner": "previous_icu_hospitalization_id",
        }
    )
    return_owner = icu_owners.rename(
        {
            "in_dttm": "return_icu_start_dttm",
            "_owner": "return_icu_hospitalization_id",
        }
    )
    events = events.join(
        previous_owner, on=[JOIN_ID, "previous_icu_start_dttm"], how="left"
    ).join(return_owner, on=[JOIN_ID, "return_icu_start_dttm"], how="left")

    raw_other = adt.filter(pl.col("std_location") == "Other").select(
        JOIN_ID, "raw_location", "in_dttm", "out_dttm"
    )
    other_by_event = (
        events.select(
            JOIN_ID,
            "icu_stay_number",
            "previous_icu_end_dttm",
            "return_icu_start_dttm",
        )
        .join(raw_other, on=JOIN_ID, how="inner")
        .filter(
            (pl.col("out_dttm") >= pl.col("previous_icu_end_dttm"))
            & (pl.col("in_dttm") <= pl.col("return_icu_start_dttm"))
        )
        .group_by([JOIN_ID, "icu_stay_number"])
        .agg(
            pl.col("raw_location")
            .unique()
            .sort()
            .alias("raw_other_categories")
        )
    )
    return (
        events.join(other_by_event, on=[JOIN_ID, "icu_stay_number"], how="left")
        .with_columns(
            (
                pl.col("previous_icu_hospitalization_id")
                != pl.col("return_icu_hospitalization_id")
            )
            .fill_null(False)
            .alias("crosses_stitched_hospitalizations")
        )
        .drop("_previous_icu_number")
    )


def _print_report(summary: pl.DataFrame, events: pl.DataFrame) -> None:
    print(f"\n{SITE_NAME}: stitched ICU readmission audit")
    print("=" * 52)
    print(
        summary.select(
            pl.len().alias("cohort_encounters"),
            pl.col(LABEL).sum().alias("cohort_positive_labels"),
            pl.col(LABEL).mean().alias("cohort_positive_rate"),
            pl.col("readmission_count").sum().alias("total_readmission_events"),
            (~pl.col("label_matches_recomputed")).sum().alias("label_mismatches"),
        )
    )
    print("\nReadmissions per stitched hospitalization:")
    print(
        summary.group_by("readmission_count")
        .agg(pl.len().alias("hospitalizations"))
        .sort("readmission_count")
    )
    print("\nReturn route families (Ward versus Other):")
    print(
        events.group_by("return_route_family")
        .agg(
            pl.len().alias("events"),
            pl.col("crosses_stitched_hospitalizations").sum().alias("cross_stitched"),
            pl.col("hours_between_icu_stays").median().alias("median_gap_hours"),
        )
        .sort("events", descending=True)
    )
    print("\nDetailed return paths:")
    print(
        events.group_by("return_path_category")
        .agg(
            pl.len().alias("events"),
            pl.col("crosses_stitched_hospitalizations").sum().alias("cross_stitched"),
            pl.col("hours_between_icu_stays").median().alias("median_gap_hours"),
        )
        .sort("events", descending=True)
    )
    print("\nRaw categories represented by Other between ICU stays:")
    print(
        events.select(
            pl.col("raw_other_categories")
            .list.explode(keep_nulls=False, empty_as_null=False)
            .alias("raw_other")
        )
        .drop_nulls()
        .group_by("raw_other")
        .agg(pl.len().alias("events"))
        .sort("events", descending=True)
    )
    print("\nPositive-label data quality:")
    print(
        summary.filter(pl.col(LABEL) == 1)
        .group_by("data_quality_status")
        .agg(pl.len().alias("hospitalizations"))
        .sort("hospitalizations", descending=True)
    )
    print("\nPositive rate by stitched member count:")
    print(
        summary.group_by("member_hospitalizations")
        .agg(
            pl.len().alias("hospitalizations"),
            pl.col(LABEL).sum().alias("positive_labels"),
            pl.col(LABEL).mean().alias("positive_rate"),
        )
        .sort("member_hospitalizations")
    )


def main() -> None:
    cfg = _read_config()
    cohort = pl.read_parquet(COHORT_PATH)
    required = {"hospitalization_id", JOIN_ID, LABEL, "split"}
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"Cohort is missing columns: {sorted(missing)}")

    cohort = cohort.with_columns(
        pl.col("hospitalization_id").cast(pl.Utf8),
        pl.col(JOIN_ID).cast(pl.Utf8),
    )
    join_ids = cohort[JOIN_ID].unique().to_list()
    idx = load_or_build_encounter_index(
        cfg, int(cfg.get("stitch_time_interval_hours", 6))
    ).with_columns(
        pl.col("hospitalization_id").cast(pl.Utf8),
        pl.col(JOIN_ID).cast(pl.Utf8),
    )
    member_ids = members_of_joins(idx, join_ids)
    raw_adt = _read_adt(cfg, member_ids)
    mapping = idx.select("hospitalization_id", JOIN_ID).unique("hospitalization_id")
    adt = (
        raw_adt.join(mapping, on="hospitalization_id", how="left")
        .with_columns(pl.col(JOIN_ID).fill_null(pl.col("hospitalization_id")))
        .filter(pl.col(JOIN_ID).is_in(join_ids))
        .rename({"hospitalization_id": RAW_HOSP_ID})
    )
    adt = _standardize_locations(adt)

    encounter_input = adt.select(
        pl.col(JOIN_ID).alias("hospitalization_id"),
        "in_dttm",
        "out_dttm",
        "std_location",
    )
    first_pass, normalized = _normalized_runs(encounter_input)
    events = _add_event_provenance(_icu_events(normalized), adt)
    quality = _raw_quality(adt)

    counts = (
        normalized.rename({"hospitalization_id": JOIN_ID})
        .group_by(JOIN_ID)
        .agg(
            (pl.col("std_location") == "ICU").sum().cast(pl.Int32).alias(
                "icu_stay_count"
            )
        )
        .with_columns(
            (pl.col("icu_stay_count") - 1).clip(lower_bound=0).alias(
                "readmission_count"
            )
        )
    )
    before_counts = (
        first_pass.rename({"hospitalization_id": JOIN_ID})
        .group_by(JOIN_ID)
        .agg((pl.col("std_location") == "ICU").sum().alias("icu_runs_before_gap_removal"))
    )
    members = (
        idx.filter(pl.col(JOIN_ID).is_in(join_ids))
        .group_by(JOIN_ID)
        .agg(pl.col("hospitalization_id").n_unique().alias("member_hospitalizations"))
    )
    event_summary = events.group_by(JOIN_ID).agg(
        pl.col("return_route_family").alias("return_route_families"),
        pl.col("return_path_category").alias("return_path_categories"),
        pl.col("return_path").alias("return_paths"),
        pl.col("raw_other_categories")
        .list.explode(keep_nulls=False, empty_as_null=False)
        .unique()
        .sort()
        .alias("raw_other_categories"),
        pl.col("crosses_stitched_hospitalizations").any().alias(
            "any_return_crosses_stitched_hospitalizations"
        ),
    )

    summary = (
        cohort.join(counts, on=JOIN_ID, how="left")
        .join(before_counts, on=JOIN_ID, how="left")
        .join(_run_sequence(normalized), on=JOIN_ID, how="left")
        .join(members, on=JOIN_ID, how="left")
        .join(event_summary, on=JOIN_ID, how="left")
        .join(quality, on=JOIN_ID, how="left")
        .with_columns(
            pl.col("icu_stay_count").fill_null(0),
            pl.col("readmission_count").fill_null(0),
            pl.col("icu_runs_before_gap_removal").fill_null(0),
            pl.col("member_hospitalizations").fill_null(1),
            pl.col("any_return_crosses_stitched_hospitalizations").fill_null(False),
            pl.col("has_nonpositive_duration").fill_null(False),
            pl.col("has_overlapping_intervals").fill_null(False),
            pl.col("has_duplicate_intervals").fill_null(False),
            pl.col("has_simultaneous_locations").fill_null(False),
        )
        .with_columns(
            (
                pl.col("icu_runs_before_gap_removal") - pl.col("icu_stay_count")
            )
            .clip(lower_bound=0)
            .alias("icu_returns_removed_as_allowed_gaps"),
            (
                pl.col(LABEL)
                == (pl.col("readmission_count") > 0).cast(pl.Int32)
            ).alias("label_matches_recomputed"),
            pl.when(pl.col("has_nonpositive_duration"))
            .then(pl.lit("review_nonpositive_duration"))
            .when(pl.col("has_simultaneous_locations"))
            .then(pl.lit("review_simultaneous_locations"))
            .when(pl.col("has_overlapping_intervals"))
            .then(pl.lit("review_overlapping_intervals"))
            .when(pl.col("has_duplicate_intervals"))
            .then(pl.lit("review_duplicate_intervals"))
            .otherwise(pl.lit("clean"))
            .alias("data_quality_status"),
        )
        .sort([LABEL, "readmission_count"], descending=True)
    )
    events = (
        events.join(
            cohort.select(JOIN_ID, "hospitalization_id", LABEL, "split"),
            on=JOIN_ID,
            how="left",
        )
        .join(quality, on=JOIN_ID, how="left")
        .sort([JOIN_ID, "icu_stay_number"])
    )

    SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(SUMMARY_OUTPUT_PATH)
    events.write_parquet(EVENT_OUTPUT_PATH)
    _print_report(summary, events)
    print(f"\nPHI encounter details: {SUMMARY_OUTPUT_PATH}")
    print(f"PHI return-event details: {EVENT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
