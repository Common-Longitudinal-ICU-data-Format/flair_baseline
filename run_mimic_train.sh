#!/bin/bash
# Train all 4 MIMIC baseline models on the new (de-numbered, task4-removed) wheel.
# Resumable: per-task done-stamps in .mimic_run_stamps/ -> a relaunch skips finished
# tasks and continues. To force a fresh full retrain, delete .mimic_run_stamps/ first.
# Small tasks first so quick wins bank before the ~1h/task landmark HPO.
# Per-task wall-clock timing -> mimic_train_20260723.log
set -u
cd /Users/sudo_sage/Documents/WORK/FLAIR_PROJECT/flair_baseline
LOG=mimic_train_20260723.log
STAMP=.mimic_run_stamps
mkdir -p "$STAMP"
TASKS="extubation_failure_24h icu_readmission icu_daily_ltach icu_daily_mortality"
overall_start=$(date +%s)
echo "##### MIMIC TRAIN START $(date -u +%FT%TZ) #####" | tee -a "$LOG"
for t in $TASKS; do
  if [ -f "$STAMP/$t.done" ]; then
    echo "===== SKIP  $t (stamp present) =====" | tee -a "$LOG"; continue
  fi
  echo "===== START $t $(date -u +%FT%TZ) =====" | tee -a "$LOG"
  s=$(date +%s)
  uv run flair-baseline train --task "$t" \
    --clif-config config/clif_config_mimic.json --out . --viz >> "$LOG" 2>&1
  rc=$?
  e=$(date +%s)
  echo "===== END   $t rc=$rc elapsed=$((e-s))s ($(( (e-s)/60 ))m$(( (e-s)%60 ))s) =====" | tee -a "$LOG"
  [ "$rc" -eq 0 ] && touch "$STAMP/$t.done"
done
overall_end=$(date +%s)
tot=$((overall_end-overall_start))
echo "##### MIMIC TRAIN DONE total=${tot}s ($((tot/60))m$((tot%60))s) $(date -u +%FT%TZ) #####" | tee -a "$LOG"
