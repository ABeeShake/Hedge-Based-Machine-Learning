#!/bin/bash
#SBATCH --job-name=hbml_t1d_post
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/t1d_post_%j.out
#SBATCH --error=slurm_logs/t1d_post_%j.err

# ==============================================================================
# Pre-registered T1D replication (reviews/preregistration_T1D_replication.md):
# post-processing, locked per-cohort evaluation and the pooled analysis.
#
#   1. Stops if any participant forecast is missing or stale (resubmit the sim array first).
#   2. For each of azt1d, ohiot1dm, brist1d, brist1d_15min:
#        run_postprocess_chunk.py --all, then the locked evaluate_external_cohort.py;
#        results are moved to external_results/<dataset>/.
#   3. pool_external_cohorts.py over the three primary cohorts (analysis P2)
#      -> external_results/pooled_t1d/pooled_statistical_tests.csv
#
# Submit after the simulation array:
#   SIM=$(sbatch --parsable slurm/run_t1d_replication_sim_array.slurm)
#   sbatch --dependency=afterok:$SIM slurm/run_t1d_replication_post.slurm
# Safe to rerun: each step overwrites its own outputs.
# ==============================================================================

set -euo pipefail
mkdir -p slurm_logs
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export FAST_ENGINE_DEVICE="cpu"

PY="./hedge-env/bin/python3"
[ -x "$PY" ] || PY="python3"

DATASETS=(azt1d ohiot1dm brist1d brist1d_15min)

echo "=============================================================================="
echo "SLURM Job ID:       ${SLURM_JOB_ID:-}"
echo "Start time:         $(date)"
echo "=============================================================================="

echo "=== [1/3] Checking that every participant forecast is complete ==="
"$PY" - "${DATASETS[@]}" <<'PYEOF'
import sys
from run_simulation_chunk import build_condition_registry, get_dataset_csv_files, get_patient_id, is_patient_complete
missing = []
for ds in sys.argv[1:]:
    for cond in build_condition_registry(datasets=[ds]):
        for f in get_dataset_csv_files(ds):
            pid = get_patient_id(f)
            if not is_patient_complete(cond["output_dir"], pid, cond=cond):
                missing.append(f"{ds} h={cond['horizon_str']} ctx={cond['context_str']} pid={pid}")
    print(f"{ds}: checked")
if missing:
    print(f"{len(missing)} participant forecasts are missing or stale, e.g.:")
    print("\n".join("  " + m for m in missing[:20]))
    print("Resubmit slurm/run_t1d_replication_sim_array.slurm (completed participants are skipped).")
    sys.exit(1)
print("All participant forecasts complete.")
PYEOF

echo "=== [2/3] Per-cohort post-processing and locked evaluation ==="
for DS in "${DATASETS[@]}"; do
    echo "--- $DS: post-processing ($(date)) ---"
    "$PY" run_postprocess_chunk.py --dataset "$DS" --all --no_plot_ceg
    echo "--- $DS: evaluate_external_cohort.py ($(date)) ---"
    "$PY" evaluate_external_cohort.py --dataset "$DS"
    mkdir -p "external_results/${DS}"
    mv -f Outputs/external/"${DS}"/* "external_results/${DS}/"
    echo "Moved results to external_results/${DS}/"
done

echo "=== [3/3] Pooled analysis over the primary cohorts (azt1d, ohiot1dm, brist1d) ==="
"$PY" pool_external_cohorts.py --cohorts azt1d,ohiot1dm,brist1d

echo "=============================================================================="
echo "T1D replication post-processing completed at $(date)"
echo "=============================================================================="
