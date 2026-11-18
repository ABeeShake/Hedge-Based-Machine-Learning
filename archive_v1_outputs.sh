#!/bin/bash
# archive_v1_outputs.sh -- move engine-v1 simulation outputs out of the way before the v2 re-run.
#
# Moves Outputs/<dataset>/h-* (forecasts, losses, weights, regrets, ceg, rmse) to
# Outputs_v1/<dataset>/ and the pooled rmse.csv / maxae.csv files alongside them, so v2 results
# never mix with v1 files. Raw data (cgmacros/, weinstock/, Outputs/<dataset>/data*) is untouched.
set -euo pipefail
cd "$(dirname "$0")"
ARCHIVE_BASE="Outputs_v1"
if [ -e "$ARCHIVE_BASE" ]; then
    ARCHIVE_BASE="Outputs_archive/$(date +%Y%m%d_%H%M%S)"
fi
for ds in weinstock cgmacros; do
    [ -d "Outputs/$ds" ] || continue
    mkdir -p "$ARCHIVE_BASE/$ds"
    for d in Outputs/$ds/h-*; do
        [ -e "$d" ] || continue
        mv "$d" "$ARCHIVE_BASE/$ds/"
    done
    for f in rmse.csv maxae.csv; do
        [ -f "Outputs/$ds/$f" ] && mv "Outputs/$ds/$f" "$ARCHIVE_BASE/$ds/$f"
    done
done
for f in rmse.csv maxae.csv full_ceg.csv; do
    [ -f "$f" ] && mkdir -p "$ARCHIVE_BASE" && mv "$f" "$ARCHIVE_BASE/$f"
done
echo "Prior engine outputs moved to $ARCHIVE_BASE/."
