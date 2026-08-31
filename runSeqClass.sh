#!/bin/bash


python scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py \
    --systems 1500 \
    --propMin 30 \
    --orbit leo \
    --OE \
    --energy \
    --transformer \
    --cnn \
    --xgboost \
    --catboost \
    --rf \
    --extratrees \
    --save \
    --loss-scheme inverse \
    --oversample \
    --loss-scheme effective \
    # --cb-beta 0.5 \
    # --test vleo \
    # --testSys 1500 \