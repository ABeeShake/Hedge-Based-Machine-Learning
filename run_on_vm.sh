#!/bin/bash
# run_on_vm.sh -- the steps that need the full simulation outputs (Outputs/ on the VM).
#
# Run from the HedgeProject root on the VM after pulling this commit:
#     bash run_on_vm.sh            # everything
#     bash run_on_vm.sh --no-push  # do not commit/push the overleaf folder
#
# Prerequisite: the engine-v2 simulations are finished (see RUN_V2.md):
#     python3 run_simulation_chunk.py --list_conditions   -> every condition DONE
# Order matters: every table must exist before generate_results_macros.py reads it.
#  0. engine checks (tests/test_engine.py), completeness check, 5-minute grid summary (analyze_segments.py)
#  1. process_results.py   re-aggregates every condition (adds HBML-FS, writes rmse.csv / ceg.csv)
#  2. compare_fixed_share.py   Fixed-Share vs current algorithm on ALL 24 conditions
#  3. generate_figures.py --rmse_only (median, then mean)  -> rmse_pivot_*, rmse_participant.csv
#  4. calculate_statistics.py   (now reads raw participant RMSE, not rounded tables)
#  5. calculate_baselines.py    (persistence, running mean, uniform; no hard-coded values)
#  6. generate_figures.py       (Clarke tables, subgroup tables, figures)
#  7. summarize_engine_timings.py (timings table from the simulation metadata)
#     secondary analyses: evaluate stages, if their simulations exist
#  8. generate_results_macros.py + check_macros.py
set -euo pipefail
PUSH=true
for a in "$@"; do [ "$a" = "--no-push" ] && PUSH=false; done

if   [ -x ./hedge-env/bin/python3 ]; then PY=./hedge-env/bin/python3
elif [ -n "${VIRTUAL_ENV:-}" ];      then PY="$VIRTUAL_ENV/bin/python"
else PY=$(command -v python3); fi
echo "Python: $PY"
mkdir -p output_logs

echo "== 0 engine checks =="
"${PY}" tests/test_engine.py
if "${PY}" run_simulation_chunk.py --list_conditions | grep -q "PENDING\|IN PROGRESS"; then
    "${PY}" run_simulation_chunk.py --list_conditions | tail -30
    echo "Some conditions are not finished with the current engine; finish the simulations first (RUN_V2.md)." >&2
    exit 1
fi

echo "== 0b 5-minute grid: gaps, segments, forecast coverage =="
"${PY}" analyze_segments.py | tee output_logs/segments.log

echo "== 1 post-processing all conditions =="
"${PY}" run_postprocess_chunk.py --all 2>&1 | tee output_logs/postprocess.log

echo "== 2 RMSE tables =="
"${PY}" generate_figures.py --rmse_only --median
"${PY}" generate_figures.py --rmse_only

echo "== 3 Aggregator and Fixed-Share comparison =="
"${PY}" compare_fixed_share.py --overwrite --drop-mismatched 2>&1 | tee output_logs/fs_comparison.log

echo "== 4 paired statistics =="
"${PY}" calculate_statistics.py

echo "== 5 baselines =="
"${PY}" calculate_baselines.py

echo "== 6 Clarke tables, subgroup tables, figures =="
"${PY}" generate_figures.py
[ -f plot_subgroups.py ] && "${PY}" plot_subgroups.py || true
[ -f copy_overleaf_images.sh ] && bash copy_overleaf_images.sh || true

echo "== 6b Clinical hypoglycemia events, weight diagnostics, and regime tables =="
"${PY}" evaluate_clinical_and_weights.py
"${PY}" analyze_pool_mechanism.py --full-only || true   # leave-one-expert-out, fixed convex weights (review point 7)
"${PY}" analyze_gamma_sensitivity.py || true
if [ -f compute_regimes.py ] && [ -f analyze_regimes.py ] && [ -f plot_regime_analysis.py ]; then
    "${PY}" compute_regimes.py --cgmacros_dir ./cgmacros --weinstock_dir ./weinstock --out_dir ./Outputs/regimes
    for ds in weinstock cgmacros; do
        for h in 0.5 2 5; do
            "${PY}" analyze_regimes.py --dataset "$ds" --data_dir "./$ds" --regimes_dir "./Outputs/regimes/$ds" \
                --forecast_root "./Outputs/$ds" --horizon "$h" --context full --out_dir ./Outputs/regime_analysis
            "${PY}" plot_regime_analysis.py --results_dir ./Outputs/regime_analysis --tag "${ds}_h${h}_cfull" \
                --out_dir ./overleaf/images/regimes
            # Keep hhalf alias updated for LaTeX figures
            if [ "$h" = "0.5" ] && [ -f "./overleaf/images/regimes/fig3_stacked_freq_${ds}_h0.5_cfull.pdf" ]; then
                cp "./overleaf/images/regimes/fig3_stacked_freq_${ds}_h0.5_cfull.pdf" \
                   "./overleaf/images/regimes/fig3_stacked_freq_${ds}_hhalf_cfull.pdf"
            fi
        done
    done
fi

echo "== 7 timings, secondary analyses, and runtime environment =="
"${PY}" summarize_engine_timings.py
"${PY}" record_runtime_env.py    # package versions + CPU/GPU/RAM -> runtime_env.csv -> \EnvHardware
[ -d Outputs/secondary/population ] && "${PY}" secondary_population_models.py --stage evaluate || echo "(no population-model outputs yet)"
[ -d Outputs/secondary/cadence ] && "${PY}" secondary_retrain_cadence.py --stage evaluate || echo "(no refit-cadence outputs yet)"
[ -d Outputs/secondary/seeds ] && "${PY}" secondary_seed_stability.py --stage evaluate || echo "(no seed-stability outputs yet)"

"${PY}" plot_main_figures.py   # main-text figures replacing the dense results tables

echo "== 8 macros =="
"${PY}" generate_results_macros.py
"${PY}" check_macros.py || echo ">>> check_macros.py reported problems (see above). Fix them before submitting."

if $PUSH && [ -d overleaf/.git ]; then
    git -C overleaf add tables/ sections/generated_macros.tex images/ 2>/dev/null || true
    git -C overleaf commit -m "Regenerate tables and macros [$(date '+%Y-%m-%d %H:%M')]" || true
    git -C overleaf push origin main || echo "Push failed; push overleaf/ manually."
fi
echo "Done. Fixed-Share report: fs_comparison/REPORT.md"
