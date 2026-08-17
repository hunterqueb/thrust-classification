#!/bin/bash

# -- latest!

# generate latex tables for the RA/Dec (angles-only, rates) ground-station frame results.
# RA/Dec has no cart/OE split (it's a single --frame radec representation), so unlike
# generateLaTeXTablesColor.sh these calls use --include-cart instead of --combine-features:
# every RA/Dec row falls into generateLatexTableCompact.py's "Cartesian" bucket (its is_oe_log()
# heuristic looks for an "OE" token in the log stem, which RA/Dec logs never have), so
# --include-cart is what actually emits a full table instead of the near-random-Cartesian summary
# paragraph. Each call also writes an (always-empty, harmless) "% No OE rows found..." _oe.tex --
# expected. Note RA/Dec is only 4 channels (RA, Dec, dRA, dDec) vs. AER/ECI/OE's 6, so these
# results aren't directly comparable feature-count-wise to the other tables.

# results only for vleo
  python gmat/data/classification/generateLatexTableCompact.py \
      --csv gmat/data/classification/parsed_data/vleo/_group/csv/summary_group.csv --train-label VLEO \
      --out-prefix class_vleo_radec --metrics rpf1 --include-cart

# results only for leo
  python gmat/data/classification/generateLatexTableCompact.py \
      --csv gmat/data/classification/parsed_data/leo/_group/csv/summary_in_group.csv --train-label LEO \
      --out-prefix class_leo_radec --metrics rpf1 --include-cart

# results for geo
  python gmat/data/classification/generateLatexTableCompact.py \
      --csv gmat/data/classification/parsed_data/geo/_group/csv/summary_group.csv --train-label GEO \
      --out-prefix class_geo_radec --metrics rpf1 --include-cart

# combine leo-to-vleo and combined-to-vleo OOD results to compactly show all RA/Dec results for the vleo test set
  python gmat/data/classification/generateLatexTableCompact.py \
      --csv gmat/data/classification/parsed_data/leo/_group/csv/summary_group.csv --train-label LEO \
      --csv gmat/data/classification/parsed_data/combined/leo-meo-geo/_group/csv/summary_group.csv  --train-label LEO-MEO-GEO \
      --out-prefix class_leo_combined_radec --metrics rpf1 --include-cart
