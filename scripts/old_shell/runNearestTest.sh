#!/bin/bash

python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py --OE --norm --energy --classic --nearest --test leo --save

python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py --OE --norm --energy --classic --nearest --test geo --save

python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py --OE --norm --energy --classic --nearest --test leo --propMin 3 --save

python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py --OE --norm --energy --classic --nearest --test geo --propMin 3 --save

