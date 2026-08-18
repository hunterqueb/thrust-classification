#!/bin/bash

# -- latest!

# Each phase below writes its logs into the same gmat/data/classification/<orbit>/ tree, and
# displayLogData.py's --group-dir mode aggregates every *.log found recursively under a
# directory with no awareness of which phase produced it. removeClassLogs.sh between phases is
# what keeps each phase's summary CSVs (and therefore its LaTeX tables) scoped to just that
# phase's runs -- do not reorder/remove those wipes without also addressing that.
#
# Each phase's generateLaTeXTablesColor*.sh writes its *.tex files into the repo root (cwd), so
# move_tables snapshots and clears them into a per-frame subfolder immediately after each phase
# runs, before the next phase's tables can land on top of them. Everything from this run lands
# under gmat/data/tables/<timestamp>/{eci,aer,radec,energy}/ so results from different runs never
# clobber each other and each frame's tables stay separated by construction rather than by
# filename convention.

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="gmat/data/tables/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"/{eci,aer,radec,energy}

move_tables () {
    local dest="$1"
    if compgen -G "*.tex" > /dev/null; then
        mv *.tex "$dest"/
    fi
}

# remove all classification results to avoid confusion
./removeClassResults.sh
# run generateThrustClass.sh to generate the classification results without energy features
./generateThrustClass.sh
# generate latex tables for results without energy features
./generateLaTeXTablesColor.sh
move_tables "$RESULTS_DIR/eci"
# remove classification logs to avoid mixing with the next phase's logs, but preserve latex table files
./removeClassLogs.sh
# run generateThrustClass_aer.sh to generate the classification results for the AER (Az/El/Range/rates) ground-station frame
./generateThrustClass_aer.sh
# generate latex tables for AER results
./generateLaTeXTablesColor_aer.sh
move_tables "$RESULTS_DIR/aer"
# remove classification logs to avoid mixing with the next phase's logs, but preserve latex table files
./removeClassLogs.sh
# run generateThrustClass_radec.sh to generate the classification results for the RA/Dec (angles-only, rates) ground-station frame
./generateThrustClass_radec.sh
# generate latex tables for RA/Dec results
./generateLaTeXTablesColor_radec.sh
move_tables "$RESULTS_DIR/radec"
# remove classification logs to avoid issue with energy features logs but preserve latex table files
./removeClassLogs.sh
# run generateThrustClass_energy.sh to generate the classification results with energy features
./generateThrustClass_energy.sh
# generate latex tables for results with energy features
./generateLaTeXTablesColor_energy.sh
move_tables "$RESULTS_DIR/energy"

echo "Results written to $RESULTS_DIR"