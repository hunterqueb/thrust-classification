#!/bin/bash
# runManuscriptInSeq.sh with --norm instead of --standardize. Logs get Norm_ in the stem, next to
# (never over) the standardize logs. Every runManuscriptInSeq.sh env var still applies, e.g.
#   ARMS="leo:geo geo:leo" ./runManuscriptInSeqNorm.sh
SCALING=norm exec "$(dirname "$0")/runManuscriptInSeq.sh" "$@"
