#!/bin/sh
# Both manuscript sweeps with a 0.1 train split (val 0.1, test 0.8). Logs get Train_600_ in the stem,
# next to (never over) the 0.7 runs, and are left out of the manuscript tables.
#
# End to end: sweeps -> parse -> in-sequence CIs -> combined tables in
# gmat/data/seqClassification/$OUT_DIR. Resumable: the sweeps skip finished cells.
#   OUT_DIR  tables dir, relative to gmat/data/seqClassification or absolute (default manuscript_tables_low)
#   N_BOOT   bootstrap replicates for the CIs (default 2000)
# Every other knob (SEEDS, PROP_MINS, FEATS, ARMS, EXTRA, ...) passes through to both sweeps.
#
# Smoke test -- OnePass_ logs, which every aggregator drops, and tables to a scratch dir so
# manuscript_tables_low is untouched:
#   SEEDS=0 PROP_MINS=10 FEATS=phys ARMS=leo:leo EXTRA=--one-pass EXTRA_TOK=OnePass_ \
#     OUT_DIR=/tmp/low_smoke N_BOOT=50 ./runManuscriptExperiment_low.sh
set -eu
export TRAIN_RATIO=0.1

PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-manuscript_tables_low}"
N_BOOT="${N_BOOT:-2000}"
ARMS="${ARMS:-leo:leo meo:meo geo:geo leo:geo geo:leo leo:meo meo:leo}"
SUBSET="Train_$(awk -v r="$TRAIN_RATIO" -v n="${SYSTEMS:-1500}" 'BEGIN { printf "%d", 4 * r * n }')"
case "$OUT_DIR" in
    /*) ;;
    *)  OUT_DIR="$PWD/gmat/data/seqClassification/$OUT_DIR" ;;
esac

./runManuscriptTotal.sh
./runManuscriptInSeq.sh

# The sweeps skip their own parse step when TRAIN_RATIO != 0.7. Group CSVs are rebuilt from every log
# under the train orbit's directory; --subset below picks the Train_600 runs back out.
ORBITS=$(echo "$ARMS" | tr ' ' '\n' | cut -d: -f1 | sort -u)
(cd gmat/data/classification && for t in $ORBITS; do
    "$PYTHON" displayLogData.py . --group-dir "$t/" --group-name manuscript
done)
(cd gmat/data/seqClassification && for t in $ORBITS; do
    "$PYTHON" displaySeqLogData.py . --group-dir "$t/" --group-name manuscript
done)

# In Sequence CIs from the saved scores (the Total Sequence ones come straight from the logs).
"$PYTHON" scripts/two_body/uncertaintyReport.py --task seq --subset "$SUBSET" --n-boot "$N_BOOT" \
    --out-dir "$OUT_DIR"
(cd gmat/data/seqClassification &&
    "$PYTHON" aggregateManuscriptCombined.py --ci --subset "$SUBSET" --n-boot "$N_BOOT" --out-dir "$OUT_DIR")
