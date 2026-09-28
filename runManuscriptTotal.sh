#!/bin/bash
#
# Manuscript sweep, WHOLE-TRAJECTORY ("total") classification: one label per trajectory, the
# counterpart to runManuscriptSweep.sh's per-timestep in-sequence sweep, with the same six models:
# Bi-LSTM, Bi-Mamba, Transformer, InceptionTime (CNN), LightGBM, MiniRocket.
# 5 arms x 3 feature sets x 3 windows x 3 seeds = 135 runs.
# Resumable: a cell whose log already carries the completion marker is skipped.
#
# Same three incremental feature sets as runManuscriptSweep.sh (see its header for why the ladder
# is kept separate from phys):
#   eci    raw ECI, no feature flags                                          (6 ch)
#   phys   + OE + J2-inclusive energy LEVEL                                   (8 ch)
#   ladder + hierarchical residual decomposition, energy RATES at 3 rungs     (11 ch)
#
# --seq-data loads through the in-sequence script's own loader (scripts/two_body/seqData.py): same
# 1500-IC datasets, same features, same train-only --standardize, and the SAME IC split for a given
# --seed. So every cell here pairs trajectory-for-trajectory with the matching runManuscriptSweep.sh
# cell; only the label changes (one class per trajectory instead of one per timestep).
#
# MUST run from the repo root: the training script opens data.yaml relatively.
#
# Smoke test:
#   DRY_RUN=1 ./runManuscriptTotal.sh | grep 'would write' | sort | uniq -d   # must be empty
#   SEEDS=0 PROP_MINS=10 FEATS=phys ARMS="leo:leo" EXTRA="--one-pass" EXTRA_TOK="OnePass_" \
#     SKIP_PARSE=1 ./runManuscriptTotal.sh
#
# Conformal uncertainty on every cell (temperature scaling + per-class conformal sets, shared with the
# in-sequence sweep via scripts/two_body/conformal.py):
#   CONFORMAL=0.05 ./runManuscriptTotal.sh
# --conformal adds no strAdd token, so the stem is unchanged; instead the completion marker moves
# to the CNN's conformal block, and every existing log without one re-runs.

set -euo pipefail

SYSTEMS="${SYSTEMS:-1500}"
PROP_MINS="${PROP_MINS:-10 30 100}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-leo:leo meo:meo geo:geo leo:geo geo:leo}"
FEATS="${FEATS:-eci phys ladder}"
EXTRA="${EXTRA:-}"            # smoke-test escape hatch, e.g. "--one-pass"
EXTRA_TOK="${EXTRA_TOK:-}"    # MUST mirror what EXTRA adds to strAdd, e.g. "OnePass_"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PARSE="${SKIP_PARSE:-0}"
CONFORMAL="${CONFORMAL:-0.05}"   # miscoverage alpha, 0.05 = 95% coverage; empty = off

SCRIPT=scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py

# LSTM (now Bi-LSTM), Mamba (now Bi-Mamba) and LightGBM run by default; the other three are opt-in.
# --standardize (not --norm) on every arm, no noise, default 0.7 train ratio and --eval-test, all
# to match the in-sequence sweep.
COMMON=(
    --systems "$SYSTEMS"
    --transformer
    --cnn
    --minirocket
    --standardize
    --eval-test
    --seq-data
    --save
)
# The CNN is the last model main() runs, and its conformal block prints after its inference time,
# so the marker is always the last thing a complete run writes.
MARKER="1D-CNN (InceptionTime) Inference Time"
# With --conformal --save each model's raw scores also go to scores/<stem>/ for
# scripts/two_body/uncertaintyReport.py; the CNN's file is written last, so a log whose conformal
# block predates the score dump (or whose run died while writing it) re-runs.
SCORES_LAST=""
if [ -n "$CONFORMAL" ]; then
    COMMON+=(--conformal "$CONFORMAL")
    MARKER="1D-CNN (InceptionTime) Uncertainty"
    SCORES_LAST="1D_CNN_InceptionTime.npz"
fi

FAILED=""
DONE=0
SKIPPED=0

run_one() {   # train test feat propMin seed
    local train=$1 test=$2 feat=$3 propMin=$4 seed=$5
    local feat_args=() feat_tok="" test_args=() test_tok="" evaltest_tok=""

    # strAdd token order is SeqData_, Energy_, J2Energy_, ResidLadder{n}_, OE_, Std_ -- mirror it
    # exactly or the resume check misses.
    case "$feat" in
        phys)
            feat_args=(--OE --energy --j2-energy)
            feat_tok="Energy_J2Energy_OE_" ;;
        ladder)
            feat_args=(--OE --energy --j2-energy --residual-ladder)
            feat_tok="Energy_J2Energy_ResidLadder3_OE_" ;;
        eci) ;;
        *)  echo "unknown FEATS value: $feat" >&2; return 1 ;;
    esac
    if [ "$test" != "$train" ]; then
        test_args=(--test "$test" --testSys "$SYSTEMS")
        test_tok="Test_${test}_"
    else
        evaltest_tok="EvalTest_"
    fi

    # Mirrors strAdd: SeqData, features, Std, OnePass, Test, EvalTest, Seed -- Seed always last.
    local stem="${propMin}min${SYSTEMS}SeqData_${feat_tok}Std_${EXTRA_TOK}${test_tok}${evaltest_tok}Seed${seed}"
    local log="gmat/data/classification/${train}/${propMin}min-${SYSTEMS}/${stem}.log"

    # Completion marker: the CNN is the last model main() runs, so a partial log (--save opens it
    # 'w' at startup) lacks this line and re-runs.
    local scores="gmat/data/classification/${train}/${propMin}min-${SYSTEMS}/scores/${stem}/${SCORES_LAST}"
    if [ -f "$log" ] && grep -q "$MARKER" "$log" && { [ -z "$SCORES_LAST" ] || [ -f "$scores" ]; }; then
        echo "[skip] $stem"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    echo "=========== ${train}->${test} | ${feat} | ${propMin}min | seed ${seed} | ${stem}"
    if [ "$DRY_RUN" = 1 ]; then
        echo "          would write $log"
        return 0
    fi

    if "$PYTHON" "$SCRIPT" "${COMMON[@]}" \
            --orbit "$train" --propMin "$propMin" --seed "$seed" \
            ${feat_args[@]+"${feat_args[@]}"} ${test_args[@]+"${test_args[@]}"} ${EXTRA}; then
        DONE=$((DONE + 1))
    else
        echo "[FAIL] $stem"
        FAILED="${FAILED} ${stem}"
    fi
}

# Seed outermost: a complete single-seed sweep lands first.
for seed in $SEEDS; do
  for propMin in $PROP_MINS; do
    for feat in $FEATS; do
      for arm in $ARMS; do
        run_one "${arm%%:*}" "${arm##*:}" "$feat" "$propMin" "$seed"
      done
    done
  done
done

echo
echo "ran=${DONE} skipped=${SKIPPED}"
[ -n "$FAILED" ] && echo "FAILED:${FAILED}"

if [ "$SKIP_PARSE" = 1 ] || [ "$DRY_RUN" = 1 ]; then exit 0; fi

# Selective classification + conformal-rate CIs from the saved scores -> manuscript_tables/
# total_uncertainty_*.csv, which aggregateManuscriptTotal.py turns into total_t10_selective_*.
# (Its total_t9_ci_* R/P/F1 intervals come from the logged confusion matrices and need no scores.)
if [ -n "$CONFORMAL" ]; then
    "$PYTHON" scripts/two_body/uncertaintyReport.py --task total
fi

# One group CSV per TRAIN orbit (cross-regime logs live under the train orbit's directory), then
# mean +/- std over seeds -> gmat/data/classification/manuscript_tables/total_*.tex. NOT
# generateLatexTableCompact.py: it keeps the best row per (model, window), which on this sweep is a
# max over seeds, test orbits and feature arms.
cd gmat/data/classification
"$PYTHON" displayLogData.py .
for train in $(echo "$ARMS" | tr ' ' '\n' | cut -d: -f1 | sort -u); do
    "$PYTHON" displayLogData.py . --group-dir "${train}/" --group-name manuscript
done
"$PYTHON" aggregateManuscriptTotal.py --out-dir manuscript_tables
cd ../../..
