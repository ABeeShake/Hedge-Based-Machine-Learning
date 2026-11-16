#!/bin/bash
# run_on_vm.sh -- the steps that need the full simulation outputs (Outputs/ on the VM).
#
# Run from the HedgeProject root on the VM after pulling this commit:
#     bash run_on_vm.sh            # everything
#     bash run_on_vm.sh --no-push  # do not commit/push the overleaf folder
#
# Order matters: every table must exist before generate_results_macros.py reads it.
#  1. process_results.py   re-aggregates every condition (adds HBML-FS, writes rmse.csv / ceg.csv)
#  2. compare_fixed_share.py   Fixed-Share vs current algorithm on ALL 24 conditions
#  3. generate_figures.py --rmse_only (median, then mean)  -> rmse_pivot_*, rmse_participant.csv
#  4. calculate_statistics.py   (now reads raw participant RMSE, not rounded tables)
#  5. calculate_baselines.py    (persistence, running mean, uniform; no hard-coded values)
#  6. generate_figures.py       (Clarke tables, subgroup tables, figures)
#  7. generate_results_macros.py + check_macros.py
set -euo pipefail
PUSH=true
for a in "$@"; do [ "$a" = "--no-push" ] && PUSH=false; done

if   [ -x ./hedge-env/bin/python3 ]; then PY=./hedge-env/bin/python3
elif [ -n "${VIRTUAL_ENV:-}" ];      then PY="$VIRTUAL_ENV/bin/python"
else PY=$(command -v python3); fi
echo "Python: $PY"
mkdir -p output_logs

echo "== 1/7 post-processing all conditions =="
"${PY}" run_postprocess_chunk.py --all 2>&1 | tee output_logs/postprocess.log

echo "== 2/7 Fixed-Share comparison =="
"${PY}" compare_fixed_share.py --overwrite --drop-mismatched 2>&1 | tee output_logs/fs_comparison.log

echo "== 3/7 RMSE tables =="
"${PY}" generate_figures.py --rmse_only --median
"${PY}" generate_figures.py --rmse_only
"${PY}" compare_fixed_share.py --report-only --drop-mismatched

echo "== 4/7 paired statistics =="
"${PY}" calculate_statistics.py

echo "== 5/7 baselines =="
"${PY}" calculate_baselines.py

echo "== 6/7 Clarke tables, subgroup tables, figures =="
"${PY}" generate_figures.py
[ -f plot_subgroups.py ] && "${PY}" plot_subgroups.py || true
[ -f copy_overleaf_images.sh ] && bash copy_overleaf_images.sh || true

echo "== 7/7 macros =="
"${PY}" generate_results_macros.py
"${PY}" check_macros.py || echo ">>> check_macros.py reported problems (see above). Fix them before submitting."

if $PUSH && [ -d overleaf/.git ]; then
    git -C overleaf add tables/ sections/generated_macros.tex images/ 2>/dev/null || true
    git -C overleaf commit -m "Regenerate tables and macros [$(date '+%Y-%m-%d %H:%M')]" || true
    git -C overleaf push origin main || echo "Push failed; push overleaf/ manually."
fi
echo "Done. Fixed-Share report: fs_comparison/REPORT.md"
