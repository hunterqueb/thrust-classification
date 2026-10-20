#!/bin/bash
# runManuscriptTotal.sh with --norm instead of --standardize. Logs get Norm_ in the stem, next to
# (never over) the standardize logs. Every runManuscriptTotal.sh env var still applies, e.g.
#   ARMS="leo:geo geo:leo" ./runManuscriptTotalNorm.sh
SCALING=norm exec "$(dirname "$0")/runManuscriptTotal.sh" "$@"
