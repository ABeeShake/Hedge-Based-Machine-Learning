#!/bin/bash
# cgmacros-horizons.sh — Run the online forecasting simulation on CGMacros.
#
# Iterates over all three forecast horizons (30 min, 2 hr, 5 hr) and four
# context window lengths (6 hr, 12 hr, 24 hr, and full history) and calls
# run_simulation.py for each combination.
#
# Prerequisites:
#   1. Install dependencies:  pip install -r requirements.txt
#   2. Place per-patient CGMacros CSVs in ./cgmacros/
#
# Expected runtime: several hours per (horizon, context) combination on GPU;
# longer on CPU. The --log_n_steps flag enables crash recovery: re-running
# the script will resume from the last saved checkpoint.
#
# Usage: bash cgmacros-horizons.sh

set -e

for HORIZON in half 2 5; do
    case $HORIZON in
        ''|*[!0-9]*) hnum=6;;
        *) hnum=$((HORIZON * 12));;
    esac

    for CONTEXT in 72 144 288; do

        case $CONTEXT in
            ''|*[!0-9]*) cnum=$CONTEXT;;
            *) cnum=$((CONTEXT / 12));;
        esac

        mkdir -p ./Outputs/cgmacros/h-"$HORIZON"hr/context-"$cnum"hr/settings
        mkdir -p ./Outputs/cgmacros/h-"$HORIZON"hr/context-"$cnum"hr/forecasts

        echo "Running CGMacros | horizon=${HORIZON}hr | context=${cnum}hr"

        python3 ./run_simulation.py \
        --input_dir ./cgmacros/ \
        --model_dir ./Outputs/cgmacros/h-"$HORIZON"hr/context-"$cnum"hr \
        --output_dir ./Outputs/cgmacros/h-"$HORIZON"hr/context-"$cnum"hr \
        --epochs 10 \
        --horizon $hnum \
        --n_workers 0 \
        --log_n_steps 500 \
        --context_len $CONTEXT \
        > >(tee -a output_logs/cgmacros_stdout.log) 2> >(tee -a output_logs/cgmacros_stderr.log >&2);

    done
done