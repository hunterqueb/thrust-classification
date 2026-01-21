#!/bin/bash



python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py --norm --orbit leo \
	--systems 15000 --test vleo \
	--testSys 15000 --propMin 3 \
	--OE --noise --save

python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py --norm --orbit leo \
	--systems 15000 --test vleo \
	--testSys 15000 --propMin 3 \
	--noise --save

python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py --norm --orbit leo \
	--systems 15000 --test vleo \
	--testSys 15000 --propMin 10 \
	--noise --save

python scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py --norm --orbit leo \
	--systems 15000 --test vleo \
	--testSys 15000 --propMin 10 \
	--noise --OE --save
