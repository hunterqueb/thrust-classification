#!/bin/bash

# -- AER (Az/El/Range/rates) ground-station frame variant of generateThrustClass.sh
# Radar-realistic representation (leo/vleo/meo); see docs/extended_study_plan.md section 1.3.
#
# PREREQUISITE: requires aerArray{Chemical,Electric,ImpBurn,NoThrust}.npz at the same 800-system
# scale as the existing statesArray*.npz data, for each orbit/propMin combination below. As of
# this study only a small (3-system, 30-min, VLEO) validation batch has been generated via
# generateSpacecraftThrustOptGroundStation.py (GMAT-Thrust-Data) -- that needs to be run at full
# scale for vleo/leo/geo (and combined/leo-meo-geo) at 10/30/100 min before this script will work.
#
# No cart/OE split here -- AER is a single frame (--frame aer), so each orbit block has one
# variant per propMin instead of two. --noise and --mlp are dropped from the flag set: --noise
# only applies to --frame eci (Cartesian pos/vel noise doesn't map onto Az/El/Range units) and
# --mlp's PCA+Hankel path isn't implemented for --frame aer/radec yet -- both would just print a
# warning and no-op if included. --norm still applies but means per-channel z-score normalization
# fit on the training split, not the ECI/OE-specific normalization.

#in distribution VLEO
## 10 minute
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --propMin 10 --train_ratio 0.2 --frame aer --save --nearest --minirocket --transformer --cnn --xgboost --catboost --rf --extratrees
## 30 minute
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --propMin 30 --train_ratio 0.2 --frame aer --save --nearest --minirocket --transformer --cnn --xgboost --catboost --rf --extratrees
## 100 minute
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --propMin 100 --train_ratio 0.2 --frame aer --save --nearest --minirocket --transformer --cnn --xgboost --catboost --rf --extratrees

#in distribution LEO
## 10 minute
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test leo --systems 800 --testSys 800 \
    --norm --propMin 10 --train_ratio 0.2 --frame aer --save --nearest --minirocket --transformer --cnn --xgboost --catboost --rf --extratrees
## 30 minute
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test leo --systems 800 --testSys 800 \
    --norm --propMin 30 --train_ratio 0.2 --frame aer --save --nearest --minirocket --transformer --cnn --xgboost --catboost --rf --extratrees
## 100 minute
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test leo --systems 800 --testSys 800 \
    --norm --propMin 100 --train_ratio 0.2 --frame aer --save --nearest --minirocket --transformer --cnn --xgboost --catboost --rf --extratrees


# in distribution GEO
## 10 minute
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit geo --test geo --systems 800 --testSys 800 \
    --norm --propMin 10 --train_ratio 0.2 --frame aer --save --nearest --minirocket --transformer --cnn --xgboost --catboost --rf --extratrees
## 30 minute
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit geo --test geo --systems 800 --testSys 800 \
    --norm --propMin 30 --train_ratio 0.2 --frame aer --save --nearest --minirocket --transformer --cnn --xgboost --catboost --rf --extratrees
## 100 minute
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit geo --test geo --systems 800 --testSys 800 \
    --norm --propMin 100 --train_ratio 0.2 --frame aer --save --nearest --minirocket --transformer --cnn --xgboost --catboost --rf --extratrees


cd gmat/data/classification
# vleo in distribution
python displayLogData.py . --group-dir vleo/
# leo in distribution
python displayLogData.py . --group-dir leo/ --group-name in_group
# geo in distribution
python displayLogData.py . --group-dir geo/

# remove indistribution leo data to avoid confusion with out of distribution leo data
rm -rf leo/
cd ../../..

# # out of distribution

# LEO to VLEO
## 10 minute
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --propMin 10 --train_ratio 0.2 --frame aer --save --nearest --minirocket --transformer --cnn --xgboost --catboost --rf --extratrees
## 30 minute
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --propMin 30 --train_ratio 0.2 --frame aer --save --nearest --minirocket --transformer --cnn --xgboost --catboost --rf --extratrees
## 100 minute
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --propMin 100 --train_ratio 0.2 --frame aer --save --nearest --minirocket --transformer --cnn --xgboost --catboost --rf --extratrees

# LEO-MEO-GEO to VLEO
## 10 minute
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --propMin 10 --train_ratio 0.2 --frame aer --save --nearest --minirocket --transformer --cnn --xgboost --catboost --rf --extratrees
## 30 minute
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --propMin 30 --train_ratio 0.2 --frame aer --save --nearest --minirocket --transformer --cnn --xgboost --catboost --rf --extratrees
## 100 minute
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --propMin 100 --train_ratio 0.2 --frame aer --save --nearest --minirocket --transformer --cnn --xgboost --catboost --rf --extratrees

cd gmat/data/classification
# leo out of distribution to VLEO
python displayLogData.py . --group-dir leo/
# combined leo, meo, geo out of distribution to VLEO
python displayLogData.py . --group-dir combined/leo-meo-geo/
