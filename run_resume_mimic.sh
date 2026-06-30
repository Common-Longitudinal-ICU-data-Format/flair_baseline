#!/bin/zsh
# Resume MIMIC training: featurize (all 5) + train (all 5, no HPO), timed.
# Cohorts + shared MEDS + vocab.json already built — not redone here.
# MIMIC ONLY — never touches rush_*.
set -e -u
cd /Users/sudo_sage/Documents/WORK/FLAIR_PROJECT/flair_baseline

CFG=config/clif_config_mimic.json
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=logs/resume_mimic_${STAMP}.log

run() {
  local label=$1; shift
  echo "######## ${label} :: start $(date +%T) ########" | tee -a "$LOG"
  local t0=$SECONDS
  "$@" 2>&1 | tee -a "$LOG"
  local dt=$(( SECONDS - t0 ))
  echo "######## ${label} :: done in ${dt}s ($(( dt/60 ))m$(( dt%60 ))s) ########" | tee -a "$LOG"
  echo "${label}=${dt}s" >> "logs/timings_${STAMP}.txt"
}

echo "=== resume MIMIC (no-HPO) @ ${STAMP} ===" | tee "$LOG"
TOTAL0=$SECONDS
run featurize uv run flair-baseline featurize --clif-config "$CFG" --out .
run train     uv run flair-baseline train     --clif-config "$CFG" --out . --no-hpo --viz
TOTAL=$(( SECONDS - TOTAL0 ))

echo "" | tee -a "$LOG"
echo "==================== TIMINGS ====================" | tee -a "$LOG"
cat "logs/timings_${STAMP}.txt" | tee -a "$LOG"
echo "TOTAL=${TOTAL}s ($(( TOTAL/60 ))m$(( TOTAL%60 ))s)" | tee -a "$LOG"
echo "log: $LOG"
