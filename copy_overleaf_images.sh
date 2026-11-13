#!/bin/bash
# ==============================================================================
# copy_overleaf_images.sh
#
# Scans simulation output directories for Clarke Error Grids and average regret
# curves and organizes them into ./overleaf/images/clarke/ and ./overleaf/images/regret/.
#
# Works seamlessly on both local environments and the remote VM.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OVERLEAF_DIR="./overleaf/images"
mkdir -p "$OVERLEAF_DIR/clarke" "$OVERLEAF_DIR/regret"

# Determine search directory (prefer ./Outputs if it exists, otherwise .)
SEARCH_DIR="."
if [ -d "./Outputs" ]; then
    SEARCH_DIR="./Outputs"
fi

echo "Scanning for simulation images in $SEARCH_DIR..."

find "$SEARCH_DIR" -not -path "*/overleaf/*" -not -path "*/.git/*" -type d -name "images" 2>/dev/null | while read -r img_dir; do
    # Only process directories matching the pattern .../h-*hr/context-*hr/images
    if [[ "$img_dir" != *"/h-"*"hr/context-"*"hr/images"* ]]; then
        continue
    fi
    
    # Get the parent directory of images
    parent_dir=$(dirname "$img_dir")
    
    # Extract components from path (handles ./dataset/... and ./Outputs/dataset/...)
    context_part=$(basename "$parent_dir")                      # e.g., context-12hr
    horizon_part=$(basename "$(dirname "$parent_dir")")          # e.g., h-5hr
    dataset_part=$(basename "$(dirname "$(dirname "$parent_dir")")") # e.g., cgmacros or Outputs
    
    # Deal with structural differences if inside Outputs folder
    if [ "$dataset_part" = "Outputs" ] || [ "$dataset_part" = "." ]; then
        dataset_part=$(basename "$(dirname "$(dirname "$(dirname "$parent_dir")")")")
    fi
    
    # Normalize dataset name (e.g. cgmacros, weinstock)
    dataset_clean=$(echo "$dataset_part" | tr '[:upper:]' '[:lower:]')
    
    # Parse prefixes for the target filenames
    h_raw=$(echo "$horizon_part" | sed 's/h-//' | sed 's/hr//')
    c_raw=$(echo "$context_part" | sed 's/context-//' | sed 's/hr//')
    
    target_prefix="h${h_raw}_c${c_raw}_eta10"
    
    # Create target directories
    mkdir -p "$OVERLEAF_DIR/clarke/${dataset_clean}"
    mkdir -p "$OVERLEAF_DIR/regret/${dataset_clean}"
    
    # 1. Copy and rename Clarke grid PDF (also update eta1 and eta1000 aliases used in results.tex)
    for cand in \
        "$img_dir/clarke_grid_average_regrets.pdf" \
        "$img_dir/clarke_grid_${target_prefix}.pdf" \
        "$img_dir"/clarke_grid_*.pdf; do
        if [ -f "$cand" ]; then
            cp "$cand" "$OVERLEAF_DIR/clarke/${dataset_clean}/${target_prefix}.pdf"
            cp "$cand" "$OVERLEAF_DIR/clarke/${dataset_clean}/h${h_raw}_c${c_raw}_eta1.pdf"
            cp "$cand" "$OVERLEAF_DIR/clarke/${dataset_clean}/h${h_raw}_c${c_raw}_eta1000.pdf"
            echo "  ✓ Copied Clarke grid: ${dataset_clean}/${horizon_part}/${context_part} -> clarke/${dataset_clean}/${target_prefix}.pdf"
            break
        fi
    done
    
    # 2. Copy and rename average regrets PDF
    for cand in \
        "$img_dir/average_regrets_average_regrets.pdf" \
        "$img_dir/average_regrets.pdf" \
        "$img_dir/average_regrets_${target_prefix}.pdf" \
        "$img_dir"/average_regrets_*.pdf; do
        if [ -f "$cand" ]; then
            cp "$cand" "$OVERLEAF_DIR/regret/${dataset_clean}/${target_prefix}.pdf"
            echo "  ✓ Copied Regret curve: ${dataset_clean}/${horizon_part}/${context_part} -> regret/${dataset_clean}/${target_prefix}.pdf"
            break
        fi
    done
done

# If legacy ../overleaf exists, sync any tables/images into ./overleaf
if [ -d "../overleaf/tables" ]; then
    rsync -a ../overleaf/tables/ ./overleaf/tables/ 2>/dev/null || true
fi
if [ -d "../overleaf/images" ]; then
    rsync -a ../overleaf/images/ ./overleaf/images/ 2>/dev/null || true
fi

echo "Finished organizing images into ./overleaf/images/"
