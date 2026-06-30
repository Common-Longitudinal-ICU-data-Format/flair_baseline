#!/bin/zsh
# Full from-scratch MIMIC training: all 5 tasks, staged, timed per stage.
# MIMIC ONLY — never touches rush_* artifacts (explicit mimic_* names below).
set -e -u
cd /Users/sudo_sage/Documents/WORK/FLAIR_PROJECT/flair_baseline

CFG=config/clif_config_mimic.json
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=logs/train_all_mimic_${STAMP}.log

run() {  # run <label> <cmd...>
  local label=$1; shift
  echo "######## ${label} :: start $(date +%T) ########" | tee -a "$LOG"
  local t0=$SECONDS
  "$@" 2>&1 | tee -a "$LOG"
  local dt=$(( SECONDS - t0 ))
  echo "######## ${label} :: done in ${dt}s ($(( dt/60 ))m$(( dt%60 ))s) ########" | tee -a "$LOG"
  echo "${label}=${dt}s" >> "logs/timings_${STAMP}.txt"
}

echo "=== from-scratch MIMIC train @ ${STAMP} ===" | tee "$LOG"
# Clear ONLY mimic artifacts (rush is never named, never read/written).
rm -rf mimic_baseline_phi mimic_baseline_non_phi_for_upload mimic_baseline_models

TOTAL0=$SECONDS
run build-cohorts uv run flair-baseline build-cohorts --clif-config "$CFG" --out .
run build-data    uv run flair-baseline build-data    --clif-config "$CFG" --out .
run build-vocab   uv run flair-baseline build-vocab   --clif-config "$CFG" --out . --vocab-out vocab.json
run featurize     uv run flair-baseline featurize     --clif-config "$CFG" --out .
run train         uv run flair-baseline train         --clif-config "$CFG" --out . --viz
TOTAL=$(( SECONDS - TOTAL0 ))

echo "" | tee -a "$LOG"
echo "==================== TIMINGS ====================" | tee -a "$LOG"
cat "logs/timings_${STAMP}.txt" | tee -a "$LOG"
echo "TOTAL=${TOTAL}s ($(( TOTAL/60 ))m$(( TOTAL%60 ))s)" | tee -a "$LOG"
echo "log: $LOG"
