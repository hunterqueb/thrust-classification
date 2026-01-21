#!/bin/bash



python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py --OE --norm --energy --propMin 3 --systems 10000 --classic --save



python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py --OE --norm --energy --propMin 30 --systems 10000 --classic --save
python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py --OE --norm --energy --propMin 100 --systems 10000 --classic --save


