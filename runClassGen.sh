#!/bin/bash

# -- latest!

# Each phase below writes its logs into the same gmat/data/classification/<orbit>/ tree, and
# displayLogData.py's --group-dir mode aggregates every *.log found recursively under a
# directory with no awareness of which phase produced it. removeClassLogs.sh between phases is
# what keeps each phase's summary CSVs (and therefore its LaTeX tables) scoped to just that
# phase's runs -- do not reorder/remove those wipes without also addressing that.

# remove all classification results to avoid confusion
./removeClassResults.sh
# run generateThrustClass.sh to generate the classification results without energy features
./generateThrustClass.sh
# generate latex tables for results without energy features
./generateLaTeXTablesColor.sh
# remove classification logs to avoid mixing with the next phase's logs, but preserve latex table files
./removeClassLogs.sh
# run generateThrustClass_aer.sh to generate the classification results for the AER (Az/El/Range/rates) ground-station frame
./generateThrustClass_aer.sh
# generate latex tables for AER results
./generateLaTeXTablesColor_aer.sh
# remove classification logs to avoid mixing with the next phase's logs, but preserve latex table files
./removeClassLogs.sh
# run generateThrustClass_radec.sh to generate the classification results for the RA/Dec (angles-only, rates) ground-station frame
./generateThrustClass_radec.sh
# generate latex tables for RA/Dec results
./generateLaTeXTablesColor_radec.sh
# remove classification logs to avoid issue with energy features logs but preserve latex table files
./removeClassLogs.sh
# run generateThrustClass_energy.sh to generate the classification results with energy features
./generateThrustClass_energy.sh
# generate latex tables for results with energy features
./generateLaTeXTablesColor_energy.sh

mv *.tex gmat/data/tables/