#!/bin/bash
# Cross-regime arms only (leo->geo, geo->leo), --norm, both tasks: whole-trajectory then in-sequence.
# Other env vars (SEEDS, PROP_MINS, FEATS, CONFORMAL, DRY_RUN, ...) pass through to both sweeps.
set -euo pipefail
cd "$(dirname "$0")"
export ARMS="leo:geo geo:leo"
./runManuscriptTotalNorm.sh
./runManuscriptInSeqNorm.sh
