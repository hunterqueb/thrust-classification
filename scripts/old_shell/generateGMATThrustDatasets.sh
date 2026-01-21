#!/usr/bin/env bash
# Generate GMAT spacecraft thrust commands and archive the resulting .npz data sets.
# Usage: ./generateGMATSpacecraftThrusts.sh <orbit_type> <number_of_runs> <numMinsToProp> [lowerAlt] [upperAlt]

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <orbit_type> <number_of_runs> <numMinsToProp> [lowerAlt] [upperAlt]"
    exit 1
fi

orbit_type=$1       # e.g., leo, vleo, meo, geo …
num_runs=$2
numMinsToProp=$3
lowerAlt=${4:-200}
upperAlt=${5:-250}

# Path bookkeeping ── all relative to *this* script’s directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
class_base="$script_dir/gmat/data/classification"

echo "Generating GMAT spacecraft thrust commands:"
echo "  orbit       : $orbit_type"
echo "  runs        : $num_runs"
echo "  propagate   : $numMinsToProp min"
echo "  altitude    : ${lowerAlt}–${upperAlt} km"
echo

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Run the four propulsion cases
# ──────────────────────────────────────────────────────────────────────────────
#   Use a timestamp *marker* so we later know which .npz files are new.
marker="$(mktemp)"
touch "$marker"

for prop in chem elec imp none; do
    python "$script_dir/gmat/scripts/generateSpacecraftThrustOpt.py" \
        --deltaV 0.25 \
        --numRandSys "$num_runs" \
        --numMinProp "$numMinsToProp" \
        --propType "$prop" \
        --lowerAlt "$lowerAlt" \
        --upperAlt "$upperAlt" 
done

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Stash the freshly created .npz files in a dataset-specific folder
#     e.g. gmat/data/classification/vleo/10min-10000
# ──────────────────────────────────────────────────────────────────────────────
dest_dir="$class_base/$orbit_type/${numMinsToProp}min-${num_runs}"
mkdir -p "$dest_dir"

echo -e "\nArchiving new .npz files to $dest_dir …"
find "$class_base" -maxdepth 1 -type f -name '*.npz' -newer "$marker" \
     -exec mv {} "$dest_dir" \;

rm "$marker"
echo "GMAT Propagations Done."

echo "Starting Conversion to OE"
python gmat/data/classification/convertToOE.py --orbit "$orbit_type" --numRandSys "$num_runs" --propTime "$numMinsToProp" --no-plot
echo "done."

