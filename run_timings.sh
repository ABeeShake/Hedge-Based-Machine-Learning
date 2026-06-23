#!/bin/bash

files=$(find . -name "*err*.log")
AGGREGATED_CSV="all_timings.csv"

echo "Report_Index,Category,Count,Total_s,Mean_s,Std_Dev_s" > "$AGGREGATED_CSV"

# Find all .log or .txt files
for log_file in $files; do
    if [ -f "$log_file" ]; then
        echo "Processing $log_file..."
        python parse_timings.py "$log_file" --csv "temp_timing.csv"
        
        # if the script dumped valid timings, append them (skip header)
        if [ -f "temp_timing.csv" ]; then
            tail -n +2 "temp_timing.csv" >> "$AGGREGATED_CSV"
            rm "temp_timing.csv"
        fi
    fi
done

echo "Aggregated all timings into $AGGREGATED_CSV"

# Now run summarize_timings to gen the LaTeX table
python summarize_timings.py "$AGGREGATED_CSV" --output "overleaf/tables/timings_summary.csv"

echo "Done! Generated LaTeX-ready table at overleaf/tables/timings_summary.csv"
