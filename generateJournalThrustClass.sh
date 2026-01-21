#!/bin/bash

#in distribution VLEO
## 10 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --save
## 30 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --save
## 100 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --save
## 10 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --OE --save
## 30 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --OE --save
## 100 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit vleo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --OE --save

# out of distribution

# LEO to VLEO
## 10 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --save
## 30 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --save
## 100 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --save
## 10 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --OE --save
## 30 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --OE --save
## 100 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit leo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --OE --save

# LEO-MEO-GEO to VLEO
## 10 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --save
## 30 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --save
## 100 minute cart
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --save
## 10 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 10 --train_ratio 0.2 --OE --save
## 30 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 30 --train_ratio 0.2 --OE --save
## 100 minute OE
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py \
    --orbit combined/leo-meo-geo --test vleo --systems 800 --testSys 800 \
    --norm --noise --propMin 100 --train_ratio 0.2 --OE --save
