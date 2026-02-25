#!/bin/bash

# -- latest!

cd gmat/data/classification
# vleo in distribution
python displayLogData.py . --group-dir vleo/
# geo in distribution
python displayLogData.py . --group-dir geo/
# leo out of distribution to VLEO
python displayLogData.py . --group-dir leo/
# combined leo, meo, geo out of distribution to VLEO
python displayLogData.py . --group-dir combined/leo-meo-geo/

# generate latex tables

# results only for vleo
python generateLatexTable.py \
  --csv parsed_data/vleo/_group/csv/summary_group.csv \
  --train-label VLEO \
  --out-prefix class_vleo

# results for geo
python generateLatexTable.py \
  --csv parsed_data/geo/_group/csv/summary_group.csv \
  --train-label GEO \
  --out-prefix class_geo

# vleo train test, leo train with vleo test
python generateLatexTable.py \
  --csv parsed_data/vleo/_group/csv/summary_group.csv \
  --train-label VLEO \
  --csv parsed_data/leo/_group/csv/summary_group.csv \
  --train-label LEO \
  --out-prefix class_vleo_leo

# vleo train test, geo train with vleo test
python generateLatexTable.py \
  --csv parsed_data/vleo/_group/csv/summary_group.csv \
  --train-label VLEO \
  --csv parsed_data/combined/leo-meo-geo/_group/csv/summary_group.csv \
  --train-label LEO-MEO-GEO \
  --out-prefix class_vleo_combined
