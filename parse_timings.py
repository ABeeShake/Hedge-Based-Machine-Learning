import sys
import re
import csv
import argparse

def parse_timing_log(log_path, output_csv=None):
    """
    Parses a log file containing one or more 'Timing Analysis Report' blocks
    and extracts the statistics into a structured format.
    """
    extracted_data = []
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file {log_path}: {e}")
        return
        
    in_report = False
    
    # Regex to match the data lines in the table
    # Example format: Category  | Count  | Total (s)  | Mean (s) | Std Dev (s)
    data_pattern = re.compile(r"^(.*?)\s+\|\s+(\d+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s*$")
    
    report_count = 0
    
    for line in lines:
        if "=== Timing Analysis Report ===" in line:
            in_report = True
            report_count += 1
            continue
            
        if in_report and line.startswith("=" * 100):
            in_report = False
            continue
            
        if in_report:
            match = data_pattern.match(line)
            if match:
                category = match.group(1).strip()
                count = int(match.group(2))
                total_s = float(match.group(3))
                mean_s = float(match.group(4))
                std_dev_s = float(match.group(5))
                
                extracted_data.append({
                    'Report_Index': report_count,
                    'Category': category,
                    'Count': count,
                    'Total_s': total_s,
                    'Mean_s': mean_s,
                    'Std_Dev_s': std_dev_s
                })
                
    if not extracted_data:
        print(f"No timing statistics found in {log_path}")
        return
        
    print(f"Extracted {len(extracted_data)} timing records across {report_count} reports from {log_path}.")
    
    if output_csv:
        keys = ['Report_Index', 'Category', 'Count', 'Total_s', 'Mean_s', 'Std_Dev_s']
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(extracted_data)
        print(f"Successfully saved extracted data to {output_csv}")
    else:
        # If no CSV is specified, just print the results nicely
        for record in extracted_data:
            print(record)
            
    return extracted_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse timing statistics from a log file.")
    parser.add_argument("log_file", help="Path to the log file to parse.")
    parser.add_argument("--csv", help="Optional path to output the data as CSV.", default=None)
    
    args = parser.parse_args()
    parse_timing_log(args.log_file, args.csv)
