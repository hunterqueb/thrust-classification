#!/bin/bash

#in distribution VLEO
python scripts/two_body/classificationReachabilityGMATThrusts.py --propMin 10 --epochs 100 \
    --orbit vleo
python scripts/two_body/classificationReachabilityGMATThrusts.py --propMin 30 --epochs 100 \
    --orbit vleo
python scripts/two_body/classificationReachabilityGMATThrusts.py --propMin 100 --epochs 100 \
    --orbit vleo


# out of distribution
# LEO test
python scripts/two_body/classificationReachabilityGMATThrusts.py --propMin 10 --epochs 100 \
    --orbit vleo --test leo --load plots/reachability_ensemble/10min_train-vleo_lstm_ensemble_best.pt
python scripts/two_body/classificationReachabilityGMATThrusts.py --propMin 30 --epochs 100 \
    --orbit vleo --test leo --load plots/reachability_ensemble/30min_train-vleo_lstm_ensemble_best.pt
python scripts/two_body/classificationReachabilityGMATThrusts.py --propMin 100 --epochs 100 \
    --orbit vleo --test leo --load plots/reachability_ensemble/100min_train-vleo_lstm_ensemble_best.pt




#in distribution VLEO OE
python scripts/two_body/classificationReachabilityGMATThrusts.py --propMin 10 --epochs 100 \
    --orbit vleo --OE
python scripts/two_body/classificationReachabilityGMATThrusts.py --propMin 30 --epochs 100 \
    --orbit vleo --OE
python scripts/two_body/classificationReachabilityGMATThrusts.py --propMin 100 --epochs 100 \
    --orbit vleo --OE


# out of distribution
# LEO test OE
python scripts/two_body/classificationReachabilityGMATThrusts.py --propMin 10 --epochs 100 \
    --orbit vleo --test leo --load plots/reachability_ensemble/10min_train-vleo_OE_lstm_ensemble_best.pt --OE
python scripts/two_body/classificationReachabilityGMATThrusts.py --propMin 30 --epochs 100 \
    --orbit vleo --test leo --load plots/reachability_ensemble/30min_train-vleo_OE_lstm_ensemble_best.pt --OE
python scripts/two_body/classificationReachabilityGMATThrusts.py --propMin 100 --epochs 100 \
    --orbit vleo --test leo --load plots/reachability_ensemble/100min_train-vleo_OE_lstm_ensemble_best.pt --OE
