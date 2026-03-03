#!/bin/bash

# -- latest!

#in distribution VLEO
## 10 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --save --mlp --nearest --minirocket --transformer --energy
## 30 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --save --mlp --nearest --minirocket --transformer --energy
## 100 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --save --mlp --nearest --minirocket --transformer --energy
## 10 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --OE --save --mlp --nearest --minirocket --transformer --energy
## 30 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --OE --save --mlp --nearest --minirocket --transformer --energy
## 100 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --OE --save --mlp --nearest --minirocket --transformer --energy

#in distribution LEO
## 10 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test leo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --save --mlp --nearest --minirocket --transformer --energy
## 30 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test leo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --save --mlp --nearest --minirocket --transformer --energy
## 100 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test leo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --save --mlp --nearest --minirocket --transformer --energy
## 10 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test leo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --OE --save --mlp --nearest --minirocket --transformer --energy
## 30 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test leo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --OE --save --mlp --nearest --minirocket --transformer --energy
## 100 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test leo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --OE --save --mlp --nearest --minirocket --transformer --energy


# in distribution GEO
## 10 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit geo --test geo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --save --mlp --nearest --minirocket --transformer --energy
## 30 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit geo --test geo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --save --mlp --nearest --minirocket --transformer --energy
## 100 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit geo --test geo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --save --mlp --nearest --minirocket --transformer --energy
## 10 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit geo --test geo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --OE --save --mlp --nearest --minirocket --transformer --energy
## 30 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit geo --test geo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --OE --save --mlp --nearest --minirocket --transformer --energy
## 100 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit geo --test geo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --OE --save --mlp --nearest --minirocket --transformer --energy


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
## 10 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --save --mlp --nearest --minirocket --transformer
## 30 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --save --mlp --nearest --minirocket --transformer
## 100 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --save --mlp --nearest --minirocket --transformer
## 10 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --OE --save --mlp --nearest --minirocket --transformer
## 30 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --OE --save --mlp --nearest --minirocket --transformer
## 100 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --OE --save --mlp --nearest --minirocket --transformer

# LEO-MEO-GEO to VLEO
## 10 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --save --mlp --nearest --minirocket --transformer
## 30 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --save --mlp --nearest --minirocket --transformer
## 100 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --save --mlp --nearest --minirocket --transformer
## 10 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --OE --save --mlp --nearest --minirocket --transformer
## 30 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --OE --save --mlp --nearest --minirocket --transformer
## 100 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --OE --save --mlp --nearest --minirocket --transformer

cd gmat/data/classification
# leo out of distribution to VLEO
python displayLogData.py . --group-dir leo/
# combined leo, meo, geo out of distribution to VLEO
python displayLogData.py . --group-dir combined/leo-meo-geo/
