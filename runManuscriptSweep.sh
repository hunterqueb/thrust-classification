#!/bin/bash
#
# Manuscript sweep: 5 arms x 2 feature sets x 3 windows x 3 seeds = 90 runs.
# ~25 min each at 30min (CNN is ~60% of it), so budget ~40 hours. Resumable: a cell whose log
# already carries the completion marker is skipped, so Ctrl-C / reboot / OOM costs one run.
#
# Arms are "train:test". combined/leo-meo-geo is deliberately absent -- it is 93% leo (1395 leo /
# 67 geo / 38 meo) and 1390-1395 of its 1500 ICs are byte-identical to the standalone leo set at
# every propMin and in all four classes, so training on it and testing on leo leaks ~93% of the
# test trajectories. leo and geo were verified IC-disjoint, so leo<->geo transfer is clean.
# meo is in-distribution only.
#
# MUST run from the repo root: the training script opens data.yaml relatively.
#
# Smoke test before committing 40 hours:
#   DRY_RUN=1 ./runManuscriptSweep.sh | grep 'would write' | sort | uniq -d   # must be empty
#   SEEDS=0 PROP_MINS=10 FEATS=phys ARMS="leo:leo" EXTRA="--one-pass" EXTRA_TOK="OnePass_" \
#     SKIP_PARSE=1 ./runManuscriptSweep.sh

set -euo pipefail

SYSTEMS="${SYSTEMS:-1500}"
PROP_MINS="${PROP_MINS:-10 30 100}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-leo:leo meo:meo geo:geo leo:geo geo:leo}"
FEATS="${FEATS:-eci phys}"
EXTRA="${EXTRA:-}"            # smoke-test escape hatch, e.g. "--one-pass"
EXTRA_TOK="${EXTRA_TOK:-}"    # MUST mirror what EXTRA adds to strAdd, e.g. "OnePass_"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PARSE="${SKIP_PARSE:-0}"

SCRIPT=scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py

# --standardize on BOTH feature arms (not just the OE one) so the two arms differ only in the
# feature set. LightGBM is on by default -- do NOT pass --no-classic. No xgboost/catboost/rf/
# extratrees/mlp/minirocket: they roughly double wall clock for baselines the paper doesn't use.
# --eval-test so in-distribution cells report the held-out split rather than the validation split
# that early stopping and best-checkpoint restore already selected on.
COMMON=(
    --systems "$SYSTEMS"
    --mode all
    --transformer
    --cnn
    --standardize
    --loss-scheme inverse
    --eval-test
    --save
)

FAILED=""
DONE=0
SKIPPED=0

run_one() {   # train test feat propMin seed
    local train=$1 test=$2 feat=$3 propMin=$4 seed=$5
    local feat_args=() feat_tok="" test_args=() test_tok="" evaltest_tok=""

    if [ "$feat" = phys ]; then
        feat_args=(--OE --energy --j2-energy)
        feat_tok="Energy_J2Energy_OE_"          # strAdd order: Energy_, J2Energy_, OE_
    fi
    if [ "$test" != "$train" ]; then
        test_args=(--test "$test" --testSys "$SYSTEMS")
        test_tok="Test_${test}_"
    else
        # EvalTest_ is only emitted in-distribution: a cross-orbit run already evaluates on the
        # held-out split and carries Test_<orbit>_, so tagging it too would split one arm's logs
        # across two names.
        evaltest_tok="EvalTest_"
    fi

    # Mirrors strAdd exactly (training script ~:405-445). Token order: features, OnePass, Test,
    # EvalTest, Seed -- Seed always last.
    local stem="${propMin}min${SYSTEMS}${feat_tok}${EXTRA_TOK}${test_tok}${evaltest_tok}Seed${seed}"
    local log="gmat/data/seqClassification/${train}/${propMin}min-${SYSTEMS}/${stem}.log"

    # Completion marker, not mere existence: --save opens the log 'w' at startup, so an interrupted
    # run leaves a partial file that would otherwise look done. LightGBM runs last in this flag set.
    if [ -f "$log" ] && grep -q "LightGBM Cascade Inference Time" "$log"; then
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
        echo "[FAIL] $stem"        # partial log lacks the marker -> next invocation retries it
        FAILED="${FAILED} ${stem}"
    fi
}

# Seed OUTERMOST, deliberately: after ~13h you have a complete single-seed sweep -- every table
# cell populated, just no error bars -- rather than a third of the cells at full precision.
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

if [ "$SKIP_PARSE" = 1 ]; then exit 0; fi

# One group CSV per TRAIN orbit: cross-regime logs live under the train orbit's directory, since
# logLoc uses --orbit. So leo/ picks up both leo->leo and leo->geo.
#
# Deliberately NOT passing --emit-per-log: with it and without --force, process_log returns empty
# DataFrames for already-parsed logs, which process_group_dir then concats away -- a re-parse after
# adding seeds would silently emit a group CSV containing only the new logs. Omitting it sets
# emit_outputs=False and bypasses that cache entirely. Use --emit-per-log --force if you also want
# the per-log runs_*.csv (parameter counts, training time).
cd gmat/data/seqClassification
for train in $(echo "$ARMS" | tr ' ' '\n' | cut -d: -f1 | sort -u); do
    "$PYTHON" displaySeqLogData.py . --group-dir "${train}/" --group-name manuscript
done
"$PYTHON" aggregateManuscript.py --out-dir manuscript_tables
cd ../../..
