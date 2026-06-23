import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import argparse
from ExpMethods.globals import GlobalValues

# Set style for a clean, academic look
sns.set_theme(style="whitegrid")
plt.rcParams.update(GlobalValues.plot_params)
plt.rcParams["figure.figsize"] = (6, 2.5)  # Wide and short for the diagram


def load_weights(patient_id, weights_dir):
    """Loads weight data for a real patient."""
    # Assuming the weights are saved as csv with columns as models and rows as time steps
    # Construct path - trying to match pattern from fix_outputs: {id_num}_expweights_eta{eta}.csv
    # We might need to find the file if eta is unknown, or take it as arg. 
    # For now, let's assume a default pattern or search.
    
    # Actually, the file might contain weights for multiple hedge variants (c, d, fd). 
    # We probably want to visualize one of them, or `fix_outputs` saved a dict of DataFrames?
    # utils.save_data saves a dict as csv? No, likely just one dataframe or it handles dicts.
    # checking sim.py: weighted_forecast returns WT as (T+1, m) numpy array.
    # get_weighted_forecasts returns exp_weights as dict {method_name: weights_array}
    # utils.save_data likely pickles if it's a dict of arrays, or maybe it saves separate files?
    # Let's assume for this task we want to plot one specific method's weights, e.g. Hedge (d).
    # But wait, fix_outputs saves `exp_weights` which is a dict. 
    # If utils.save_data saves a dict of numpy arrays, it probably uses pickle or npz?
    # Or maybe it iterates? 
    # Let's look at `utils.save_data` usage in `fix_outputs.py`:
    # utils.save_data(exp_weights, path = os.path.join(weights_dir,f"{id_num}_expweights_eta{args.eta}.csv"))
    # If it saves to .csv, maybe it flattens it? Or maybe it expects a single dataframe?
    # If `exp_weights` is a dict of arrays like { "Hedge (c)": arr1, "Hedge (d)": arr2 }, 
    # saving that to a single CSV is non-trivial unless it's concatenated.
    
    # Let's assume for the plotting script we want to load one CSV that contains the weights 
    # for the models *within* a specific ensemble method (e.g. weights of ARIMA, NODE, etc. inside Hedge).
    # The `weighted_forecast` function returns `WT` which is (T+1, m).
    
    # User said: "Using the saved weights for a reeal patient, I would like to generate the weight plot"
    # I'll update main to take a path or ID.
    
    pass

def plot_weights(file_path, output_path=None):
    
    try:
        # Load data. 
        # Attempt to load as pickle first if extension is .pkl, else csv
        if file_path.endswith('.pkl'):
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            # Data might be the dict returned by get_weighted_forecasts
            # We need to extract one entry, e.g. "Hedge (d)"
            if isinstance(data, dict):
                 # Just pick the first key or prefer 'Hedge (d)'
                 key = "Hedge (d)" if "Hedge (d)" in data else list(data.keys())[0]
                 weights = data[key]
                 print(f"Plotting weights for method: {key}")
            else:
                weights = data
        else:
            # CSV assumption: Time x Models
            weights = pd.read_csv(file_path)
            
            # If it was saved as a dict of outcomes in one CSV (unlikely for arrays), 
            # or if it is just the raw weights matrix.
            # Let's assume it's a DataFrame where columns are Expert Names and index is Time.
            # However, `weighted_forecast` returns a numpy array. 
            # We need to know the expert names to label them.
            pass

    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # If weights is numpy array, we need column names.
    # Let's assume standard set if not provided.
    if isinstance(weights, np.ndarray):
        # Default global names or generic
        names = ['ARIMA', 'NODE', 'XGBoost', 'ETS', 'LSTM', 'TFT'] # Common ones
        # Slice or pad names to match shape
        if weights.shape[1] <= len(names):
             names = names[:weights.shape[1]]
        else:
             names = [f"Expert {i+1}" for i in range(weights.shape[1])]
    elif isinstance(weights, pd.DataFrame):
        names = weights.columns.tolist()
        weights = weights.values
        
    T = weights.shape[0]
    t = np.arange(T)
    
    # Filter only to known colors if we want to be strict, or cycle
    colors = []
    for name in names:
        if name in GlobalValues.color_params:
            colors.append(GlobalValues.color_params[name])
        else:
            colors.append(None) # Stackplot handles None? defaults to cycle

    # Handle cases where weights might not sum to 1 due to saving issues or initialization
    # Normalize just in case for stackplot
    row_sums = weights.sum(axis=1)
    weights = weights / row_sums[:, np.newaxis]

    fig, ax = plt.subplots()
    
    # Stackplot
    ax.stackplot(t, weights.T, labels=names, colors=colors, alpha=0.8)
    
    ax.set_xlim(0, T-1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time Step ($t$)')
    ax.set_ylabel('Expert Weight ($w_{k,t}$)')
    ax.set_title('Evolution of Expert Weights', fontweight='bold')
    
    # Legend outside
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot expert weights for a patient.")
    parser.add_argument("--file", type=str, required=True, help="Path to the weights file (csv or pkl).")
    parser.add_argument("--output", type=str, default="expert_weights_plot.pdf", help="Output path for the plot.")
    
    args = parser.parse_args()
    
    plot_weights(args.file, args.output)

if __name__ == "__main__":
    main()
