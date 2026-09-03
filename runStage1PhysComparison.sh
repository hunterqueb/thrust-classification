#!/bin/bash

python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 30 \
    --orbit leo \
    --cnn \
    --save \
    --loss-scheme inverse \
    --no
  --standardize --energy --OE --physics-loss-weight 0.0 --mode cascade

python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 30 \
    --orbit leo \
    --cnn \
    --save \
    --loss-scheme inverse \
  --standardize --energy --OE --physics-loss-weight 0.1 --mode cascade

python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 30 \
    --orbit leo \
    --cnn \
    --save \
    --loss-scheme inverse \
  --standardize --energy --OE --physics-loss-weight 0.2 --mode cascade

  python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 30 \
    --orbit leo \
    --cnn \
    --save \
    --loss-scheme inverse \
  --standardize --energy --OE --physics-loss-weight 0.3 --mode cascade

  python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 30 \
    --orbit leo \
    --cnn \
    --save \
    --loss-scheme inverse \
  --standardize --energy --OE --physics-loss-weight 0.4 --mode cascade

  python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 30 \
    --orbit leo \
    --cnn \
    --save \
    --loss-scheme inverse \
  --standardize --energy --OE --physics-loss-weight 0.5 --mode cascade

python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 30 \
    --orbit leo \
    --cnn \
    --save \
    --loss-scheme inverse \
  --standardize --energy --OE --physics-loss-weight 0.6 --mode cascade

python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 30 \
    --orbit leo \
    --cnn \
    --save \
    --loss-scheme inverse \
  --standardize --energy --OE --physics-loss-weight 0.7 --mode cascade

python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 30 \
    --orbit leo \
    --cnn \
    --save \
    --loss-scheme inverse \
  --standardize --energy --OE --physics-loss-weight 0.8 --mode cascade

  python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 30 \
    --orbit leo \
    --cnn \
    --save \
    --loss-scheme inverse \
  --standardize --energy --OE --physics-loss-weight 0.9 --mode cascade

python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 30 \
    --orbit leo \
    --cnn \
    --save \
    --loss-scheme inverse \
  --standardize --energy --OE --physics-loss-weight 1.0 --mode cascade