#!/bin/bash
# run_all_process_results.sh — Post-process all simulation outputs into HBML
# forecasts, regrets, and Clarke Error Grid plots.
#
# Finds every context-* output directory under ./Outputs/ and calls
# process_results.py on each one with the settings used in the paper:
#   --adaptive_eta       : AdaHedge (max-loss-based) learning rate schedule
#   --update_loss_type mse : MSE losses for the Hedge weight update
#   --norm_type ratio    : per-step ratio normalization (bounded-loss validity)
#   --advanced_methods   : also run Variable Share and other ensemble variants
#   --overwrite          : replace any existing regret/forecast files
#   --plot_ceg           : generate Clarke Error Grid PDFs
#
# Prerequisites: run cgmacros-horizons.sh and/or weinstock-horizons.sh first.
#
# Usage: bash run_all_process_results.sh

set -e

echo "Starting batch post-processing for all experimental settings..."

mkdir -p output_logs

for d in $(find ./Outputs -type d -name "context-*"); do
    dataset=$(echo "$d" | cut -d'/' -f3)

    echo "======================================================================"
    echo "Processing: $d"
    echo "Dataset:    $dataset"

    python3 process_results.py \
        --output_dir "$d" \
        --data_dir "./$dataset" \
        --adaptive_eta \
        --update_loss_type mse \
        --norm_type ratio \
        --advanced_methods \
        --overwrite \
        --plot_ceg

done

echo "======================================================================"
echo "All experimental settings processed successfully!"
