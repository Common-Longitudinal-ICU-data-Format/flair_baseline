#!/bin/bash
# Train all 4 MIMIC baseline models. Assumes `prepare` has already produced the
# cohorts, shared MEDS and features (see README) — this script only fits models.
#
# Resumable: per-task done-stamps in .mimic_run_stamps/ -> a relaunch skips finished
# tasks and continues. To force a fresh full retrain, delete .mimic_run_stamps/ first.
# Small tasks first so quick wins bank before the slow landmark tasks (~1h/task with
# HPO on; much faster with --no-hpo).
#
# Extra args are passed straight through to `flair-baseline local-training`, so:
#   ./run_mimic_train.sh --no-hpo            # skip the Optuna sweep
#   ./run_mimic_train.sh --task icu_readmission
#
# MIMIC is the source site: it produces the `local` kind only, and that bundle
# (<site>_baseline_models/<task>/local/) is what ships to other sites.
#
# Per-task wall-clock timing -> mimic_train_<UTC date>.log
set -u

# Run from the repo root regardless of where the script is invoked from. The
# previous hardcoded absolute path broke on every machine but one.
cd "$(dirname "$0")" || exit 1

CONFIG=config/clif_config_mimic.json
if [ ! -f "$CONFIG" ]; then
  echo "missing $CONFIG — copy config/clif_config.template.json and set your" >&2
  echo "site name + MIMIC data_directory (see README)." >&2
  exit 1
fi

LOG=mimic_train_$(date -u +%Y%m%d).log
STAMP=.mimic_run_stamps
mkdir -p "$STAMP"
TASKS="extubation_failure_24h icu_readmission icu_daily_ltach icu_daily_mortality"
overall_start=$(date +%s)
echo "##### MIMIC TRAIN START $(date -u +%FT%TZ) args=$* #####" | tee -a "$LOG"
for t in $TASKS; do
  if [ -f "$STAMP/$t.done" ]; then
    echo "===== SKIP  $t (stamp present) =====" | tee -a "$LOG"; continue
  fi
  echo "===== START $t $(date -u +%FT%TZ) =====" | tee -a "$LOG"
  s=$(date +%s)
  uv run flair-baseline local-training --task "$t" \
    --clif-config "$CONFIG" --out . --viz "$@" >> "$LOG" 2>&1
  rc=$?
  e=$(date +%s)
  echo "===== END   $t rc=$rc elapsed=$((e-s))s ($(( (e-s)/60 ))m$(( (e-s)%60 ))s) =====" | tee -a "$LOG"
  [ "$rc" -eq 0 ] && touch "$STAMP/$t.done"
done
overall_end=$(date +%s)
tot=$((overall_end-overall_start))
echo "##### MIMIC TRAIN DONE total=${tot}s ($((tot/60))m$((tot%60))s) $(date -u +%FT%TZ) #####" | tee -a "$LOG"
