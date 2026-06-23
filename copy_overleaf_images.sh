#!/bin/bash

# Find all images directories matches
find .. -type d -name "images" | while read -r img_dir; do
    # Only process directories matching the pattern .../h-*hr/context-*hr/images
    if [[ "$img_dir" != *"/h-"*"hr/context-"*"hr/images" ]]; then
        continue
    fi
    
    # Get the parent directory of images
    parent_dir=$(dirname "$img_dir")
    
    # Extract components from path (handles both ./dataset/... and ./Outputs/dataset/...)
    context_part=$(basename "$parent_dir")     # context-12hr
    horizon_part=$(basename $(dirname "$parent_dir")) # h-5hr
    dataset_part=$(basename $(dirname $(dirname "$parent_dir"))) # dataset
    
    # Deal with structural differences if they are inside Outputs folder
    if [ "$dataset_part" == "Outputs" ]; then
        dataset_part=$(basename $(dirname $(dirname $(dirname "$parent_dir"))))
    fi
    
    # Parse prefixes for the target filenames
    h_raw=$(echo "$horizon_part" | sed 's/h-//' | sed 's/hr//')
    c_raw=$(echo "$context_part" | sed 's/context-//' | sed 's/hr//')
    
    target_prefix="h${h_raw}_c${c_raw}_eta10"
    
    # Create target directories
    mkdir -p "./overleaf/images/clarke/${dataset_part}"
    mkdir -p "./overleaf/images/regret/${dataset_part}"
    
    # Copy and rename clarke grid
    if [ -f "$img_dir/clarke_grid_average_regrets.pdf" ]; then
        cp "$img_dir/clarke_grid_average_regrets.pdf" "./overleaf/images/clarke/${dataset_part}/${target_prefix}.pdf"
        echo "Copied ${dataset_part}/${horizon_part}/${context_part} -> clarke/${dataset_part}/${target_prefix}.pdf"
    fi
    
    # Copy and rename average regrets
    if [ -f "$img_dir/average_regrets_average_regrets.pdf" ]; then
        cp "$img_dir/average_regrets_average_regrets.pdf" "./overleaf/images/regret/${dataset_part}/${target_prefix}.pdf"
        echo "Copied ${dataset_part}/${horizon_part}/${context_part} -> regret/${dataset_part}/${target_prefix}.pdf"
    elif [ -f "$img_dir/average_regrets.pdf" ]; then
        cp "$img_dir/average_regrets.pdf" "./overleaf/images/regret/${dataset_part}/${target_prefix}.pdf"
        echo "Copied ${dataset_part}/${horizon_part}/${context_part} -> regret/${dataset_part}/${target_prefix}.pdf"
    fi
    
done

echo "Finished copying images to ./overleaf/images/"
