#!/bin/bash

# J2 Energy Test w/ OE
python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 10 \
    --orbit leo \
    --transformer \
    --cnn \
    --xgboost \
    --catboost \
    --rf \
    --extratrees \
    --save \
    --loss-scheme inverse\
    --standardize --energy --OE --j2-energy

python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 30 \
    --orbit leo \
    --transformer \
    --cnn \
    --xgboost \
    --catboost \
    --rf \
    --extratrees \
    --save \
    --loss-scheme inverse\
    --standardize --energy --OE --j2-energy

python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 100 \
    --orbit leo \
    --transformer \
    --cnn \
    --xgboost \
    --catboost \
    --rf \
    --extratrees \
    --save \
    --loss-scheme inverse\
    --standardize --energy --OE --j2-energy

# J2 Energy Test w/ ECI
python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 10 \
    --orbit leo \
    --transformer \
    --cnn \
    --xgboost \
    --catboost \
    --rf \
    --extratrees \
    --save \
    --loss-scheme inverse\
    --standardize --energy --j2-energy

python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 30 \
    --orbit leo \
    --transformer \
    --cnn \
    --xgboost \
    --catboost \
    --rf \
    --extratrees \
    --save \
    --loss-scheme inverse\
    --standardize --energy --j2-energy

python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 100 \
    --orbit leo \
    --transformer \
    --cnn \
    --xgboost \
    --catboost \
    --rf \
    --extratrees \
    --save \
    --loss-scheme inverse\
    --standardize --energy --j2-energy

# geo w/ OE and j2 energy
python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 10 \
    --orbit geo \
    --transformer \
    --cnn \
    --xgboost \
    --catboost \
    --rf \
    --extratrees \
    --save \
    --loss-scheme inverse\
    --standardize --energy --OE --j2-energy

python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 30 \
    --orbit geo \
    --transformer \
    --cnn \
    --xgboost \
    --catboost \
    --rf \
    --extratrees \
    --save \
    --loss-scheme inverse\
    --standardize --energy --OE --j2-energy

python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 100 \
    --orbit geo \
    --transformer \
    --cnn \
    --xgboost \
    --catboost \
    --rf \
    --extratrees \
    --save \
    --loss-scheme inverse\
    --standardize --energy --OE --j2-energy
