#!/bin/bash
# archive_v1_outputs.sh -- move engine-v1 simulation outputs out of the way before the v2 re-run.
#
# Moves Outputs/<dataset>/h-* (forecasts, losses, weights, regrets, ceg, rmse) to
# Outputs_v1/<dataset>/ and the pooled rmse.csv / maxae.csv files alongside them, so v2 results
# never mix with v1 files. Raw data (cgmacros/, weinstock/, Outputs/<dataset>/data*) is untouched.
set -euo pipefail
cd "$(dirname "$0")"
for ds in weinstock cgmacros; do
    [ -d "Outputs/$ds" ] || continue
    mkdir -p "Outputs_v1/$ds"
    for d in Outputs/$ds/h-*; do
        [ -e "$d" ] || continue
        if [ -e "Outputs_v1/$ds/$(basename "$d")" ]; then
            echo "Outputs_v1/$ds/$(basename "$d") already exists; not overwriting. Remove or rename it first." >&2
            exit 1
        fi
        mv "$d" "Outputs_v1/$ds/"
    done
    for f in rmse.csv maxae.csv; do
        [ -f "Outputs/$ds/$f" ] && mv "Outputs/$ds/$f" "Outputs_v1/$ds/$f"
    done
done
for f in rmse.csv maxae.csv full_ceg.csv; do
    [ -f "$f" ] && mv "$f" "Outputs_v1/$f"
done
echo "Engine-v1 outputs moved to Outputs_v1/."
