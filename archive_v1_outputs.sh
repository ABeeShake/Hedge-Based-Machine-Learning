#!/bin/bash
# archive_v1_outputs.sh -- move engine-v1 simulation outputs out of the way before the v2 re-run.
#
# Moves Outputs/<dataset>/h-* (forecasts, losses, weights, regrets, ceg, rmse) to
# Outputs_v1/<dataset>/ and the pooled rmse.csv / maxae.csv files alongside them, so v2 results
# never mix with v1 files. Raw data (cgmacros/, weinstock/, Outputs/<dataset>/data*) is untouched.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-./hedge-env/bin/python3}"
[ -x "$PY" ] || PY="python3"

# Detect engine version from existing forecast metadata in Outputs/, defaulting to 1
ENGINE_VER=$("$PY" -c '
import glob, json
files = sorted(glob.glob("Outputs/*/h-*/context-*/forecasts/*.meta.json"))
ver = "1"
for f in files[:5]:
    try:
        with open(f) as fh:
            meta = json.load(fh)
        s = meta.get("settings", meta)
        v = s.get("engine_version")
        if v:
            ver = str(v)
            break
    except Exception:
        pass
print(ver)
')

RUN_NUM=1
while [ -e "Outputs_v${ENGINE_VER}_${RUN_NUM}" ]; do
    RUN_NUM=$((RUN_NUM + 1))
done
ARCHIVE_BASE="Outputs_v${ENGINE_VER}_${RUN_NUM}"
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
