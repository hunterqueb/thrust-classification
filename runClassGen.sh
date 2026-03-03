#!/bin/bash

# -- latest!

# run generateThrustClass.sh to generate the classification results without energy features
./generateThrustClass.sh
# generate latex tables for results without energy features
./generateLatexTablesColor.sh
# remove classification logs to avoid issue with energy features logs
./removeClassLogs.sh
# run generateThrustClass_energy.sh to generate the classification results with energy features
./generateThrustClass_energy.sh
# generate latex tables for results with energy features
./generateLaTeXTablesColor_energy.sh