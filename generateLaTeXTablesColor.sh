#!/bin/bash

# -- latest!

# generate latex tables

# # results only for vleo
  python generateLatexTableCompact.py \
      --csv parsed_data/vleo/_group/csv/summary_group.csv --train-label VLEO \
      --out-prefix class_vleo --metrics rpf1 --combine-features

# results only for leo
  python generateLatexTableCompact.py \
      --csv parsed_data/leo/_group/csv/summary_in_group.csv --train-label LEO \
      --out-prefix class_leo --metrics rpf1 --combine-features

# # results for geo
  python generateLatexTableCompact.py \
      --csv parsed_data/geo/_group/csv/summary_group.csv --train-label GEO \
      --out-prefix class_geo --metrics rpf1 --combine-features

# # vleo train test, leo train with vleo test
#   python generateLatexTableCompact.py \
#       --csv parsed_data/vleo/_group/csv/summary_group.csv --train-label VLEO \
#       --csv parsed_data/leo/_group/csv/summary_group.csv  --train-label LEO \
#       --out-prefix class_vleo_leo --metrics rpf1 --oe-only

# # vleo train test, geo train with vleo test
#   python generateLatexTableCompact.py \
#       --csv parsed_data/vleo/_group/csv/summary_group.csv --train-label VLEO \
#       --csv parsed_data/combined/leo-meo-geo/_group/csv/summary_group.csv  --train-label LEO-MEO-GEO \
#       --out-prefix class_vleo_combined --metrics rpf1 --oe-only

# combine previous two tables into one table to compactly show all results for vleo test set
  python generateLatexTableCompact.py \
      --csv parsed_data/leo/_group/csv/summary_group.csv --train-label LEO \
      --csv parsed_data/combined/leo-meo-geo/_group/csv/summary_group.csv  --train-label LEO-MEO-GEO \
      --out-prefix class_leo_combined --metrics rpf1 --oe-only

cd ../../..