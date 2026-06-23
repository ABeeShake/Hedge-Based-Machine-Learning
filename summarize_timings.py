import pandas as pd
import numpy as np
import argparse

def summarize_timings(csv_file, output_file):
    df = pd.read_csv(csv_file)
    
    # Calculate overall mean and standard deviation of the 'Mean_s' column per category
    # This represents the average time per operation and its variance across different logs/reports
    summary = df.groupby("Category")["Mean_s"].agg(["mean", "std"]).reset_index()
    
    # Format for LaTeX
    summary["Time (s)"] = summary.apply(
        lambda row: f"{row['mean']:.4f} $\\pm$ {row['std']:.4f}" if pd.notna(row['std']) else f"{row['mean']:.4f}",
        axis=1
    )
    
    def parse_model(x):
        if ":" in x:
            return x.split(":")[1].strip()
        elif "HBML" in x:
            return "HBML"
        return x

    def parse_cat(x):
        if ":" in x:
            return x.split(":")[0].replace("Expert ", "").strip()
        elif "HBML" in x:
            return "Weight Update"
        return x
        
    summary["Model"] = summary["Category"].apply(parse_model)
    summary["Cat"] = summary["Category"].apply(parse_cat)
    
    summary = summary.sort_values(["Model", "Cat"])
    
    final_rows = []
    for model, group in summary.groupby("Model"):
        n_rows = len(group)
        for i, (_, row) in enumerate(group.iterrows()):
            if i == 0:
                final_rows.append({
                    "Model": f"\\multirow{{{n_rows}}}{{*}}{{{model}}}",
                    "Category": row["Cat"],
                    "Time (s)": row["Time (s)"]
                })
            else:
                final_rows.append({
                    "Model": "",
                    "Category": row["Cat"],
                    "Time (s)": row["Time (s)"]
                })
                
    final_df = pd.DataFrame(final_rows)
    final_df["Category"] = final_df["Category"].str.replace("_", "\\_", regex=False)
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"LaTeX-ready summary saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Aggregated CSV file from parse_timings.py")
    parser.add_argument("--output", default="overleaf/tables/timings_summary.csv", help="Output LaTeX-ready CSV")
    args = parser.parse_args()
    
    summarize_timings(args.input_csv, args.output)
