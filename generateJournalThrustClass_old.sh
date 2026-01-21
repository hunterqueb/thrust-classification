#!/bin/bash

#in distribution VLEO
## 3 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 3 --train_ratio 0.2 --save
## 10 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --save
## 3 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 3 --train_ratio 0.2 --OE --save
## 10 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --OE --save

# out of distribution

# LEO to VLEO
## 3 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 3 --train_ratio 0.2 --save
## 10 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --save
## 3 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 3 --train_ratio 0.2 --OE --save
## 10 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --OE --save

# LEO-MEO-GEO to VLEO
## 3 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 3 --train_ratio 0.2 --save
## 10 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --save
## 3 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 3 --train_ratio 0.2 --OE --save
## 10 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --OE --save