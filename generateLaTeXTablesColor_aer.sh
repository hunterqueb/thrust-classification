#!/bin/bash

# -- latest!

# generate latex tables for the AER (Az/El/Range/rates) ground-station frame results.
# AER has no cart/OE split (it's a single --frame aer representation), so unlike
# generateLaTeXTablesColor.sh these calls use --include-cart instead of --combine-features:
# every AER row falls into generateLatexTableCompact.py's "Cartesian" bucket (its is_oe_log()
# heuristic looks for an "OE" token in the log stem, which AER logs never have), so --include-cart
# is what actually emits a full table instead of the near-random-Cartesian summary paragraph.
# Each call also writes an (always-empty, harmless) "% No OE rows found..." _oe.tex -- expected.

# results only for vleo
  python gmat/data/classification/generateLatexTableCompact.py \
      --csv gmat/data/classification/parsed_data/vleo/_group/csv/summary_group.csv --train-label VLEO \
      --out-prefix class_vleo_aer --metrics rpf1 --include-cart

# results only for leo
  python gmat/data/classification/generateLatexTableCompact.py \
      --csv gmat/data/classification/parsed_data/leo/_group/csv/summary_in_group.csv --train-label LEO \
      --out-prefix class_leo_aer --metrics rpf1 --include-cart

# results for geo
  python gmat/data/classification/generateLatexTableCompact.py \
      --csv gmat/data/classification/parsed_data/geo/_group/csv/summary_group.csv --train-label GEO \
      --out-prefix class_geo_aer --metrics rpf1 --include-cart

# combine leo-to-vleo and combined-to-vleo OOD results to compactly show all AER results for the vleo test set
  python gmat/data/classification/generateLatexTableCompact.py \
      --csv gmat/data/classification/parsed_data/leo/_group/csv/summary_group.csv --train-label LEO \
      --csv gmat/data/classification/parsed_data/combined/leo-meo-geo/_group/csv/summary_group.csv  --train-label LEO-MEO-GEO \
      --out-prefix class_leo_combined_aer --metrics rpf1 --include-cart
