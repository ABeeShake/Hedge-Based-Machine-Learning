"""run_simulation.py — Per-patient online forecasting simulation.

This is the primary entry point for running the HBML glucose forecasting
experiments. For each patient CSV in the specified input directory it runs a
fully sequential, personalized simulation in which expert models generate
forecasts at every time step and their weights are updated online.

Usage
-----
    python run_simulation.py \\
        --input_dir  <path/to/patient/csvs> \\
        --output_dir <path/to/save/forecasts> \\
        --model_dir  <path/to/save/model/checkpoints> \\
        --horizon    <forecast horizon in 5-min steps, e.g. 6 = 30 min> \\
        --context_len <rolling context window in steps, e.g. 288 = 24 hr> \\
        --epochs     <max training epochs per retrain> \\
        --n_workers  <DataLoader worker count; use 0 on macOS/CPU> \\
        --log_n_steps <checkpoint interval>

Key CLI flags
-------------
--input_dir   Directory containing per-patient CSV files.
--output_dir  Root directory for forecasts/, settings/, and model checkpoints.
--model_dir   Directory from which to load pre-trained model weights
              (warm-start); defaults to --output_dir.
--horizon     Forecast horizon in 5-minute steps (6=30 min, 24=2 hr, 60=5 hr).
--context_len Number of most-recent observations fed to each expert model
              (-1 = full available history).
--epochs      Maximum training epochs per periodic retrain (default 50).
--log_n_steps Save forecast CSVs every N steps, enabling resume after crash.
--debug       Truncate each patient to the first 100 steps for quick testing.

Output layout
-------------
    <output_dir>/
        forecasts/<patient_id>_forecasts.csv   per-step expert forecasts
        settings/<patient_id>.json             simulation configuration
        node/<patient_id>_node_iteration*.pt   NODE model checkpoints

See also
--------
run_all_process_results.sh  — post-process forecasts into HBML weights/regrets
generate_figures.py         — produce paper figures from processed results
"""
import os

import argparse
import numpy as np
import pandas as pd
import torch
import lightning as L

import ExpMethods.models as m
import ExpMethods.data as data
import ExpMethods.utils as utils
import ExpMethods.simulate as sim
import ExpMethods.visualizations as viz

from glob import glob
from ExpMethods.globals import GlobalValues
from ExpMethods.timing import Timer, TimingRegistry
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint 
from statsforecast.models import AutoETS, AutoARIMA 
from xgboost import XGBRegressor
from neuralforecast.models import NHITS

def main():
    
    args = get_args()
    
    raise_errors(args)
    
    torch.set_float32_matmul_precision('high')
    
    max_horizon = args.horizon
    max_batch_size = args.batch
    max_epochs = args.epochs
    tol = args.tolerance
    t_start = args.t_start
    
    trainer_params = sim.DefaultSimulationParams.trainer_params(max_epochs = max_epochs)
    
    trainer = L.Trainer(**trainer_params)
    
    dir_params = dict(
        input_dir = args.input_dir,
        output_dir = args.output_dir,
        model_dir = args.model_dir)
 
    sim_params = sim.DefaultSimulationParams.sim_params(
        horizon = max_horizon,
        batch_size = max_batch_size,
        epochs = max_epochs,
        output_dir = args.output_dir,
        num_workers = args.n_workers,
        log_n_steps = args.log_n_steps,
        context_len = args.context_len)

    node_params = m.DefaultModelParams.node_params(
        horizon = max_horizon)
        
    xgboost_params = m.DefaultModelParams.xgboost_params()
    
    nhits_params = m.DefaultModelParams.nf_params(
        omit = ["train_n_steps", "hidden_size"],
        input_size = args.context_len,
        h = max_horizon
        )
        
    autoformer_params = m.DefaultModelParams.nf_params(
        omit = ["train_n_steps", "mlp_units"],
        input_size = args.context_len,
        h = max_horizon,
        learning_rate = 0.1
        )
    base_model_dict = {
        "NODE": m.NODE(**node_params),
        "XGBoost": XGBRegressor(**xgboost_params),
        "NHITS": NHITS(**nhits_params)
        }
   
    for path in glob(os.path.join(args.input_dir,"*.csv")):

        id_num = os.path.basename(path)[os.path.basename(path).find("-") + 1: os.path.basename(path).find("-") + 4]
        
        existing_forecasts = os.path.join(args.output_dir, "forecasts", f"{id_num}_forecasts.csv")
        
        if os.path.exists(existing_forecasts):
            with open(existing_forecasts,"rb") as f:
                num_lines = sum(1 for _ in f)
            # num_lines includes header. Last data row is at CSV row (num_lines-1).
            # Due to 0-indexing, this is forecast index (num_lines-2).
            # That forecast was made at timestep t = (num_lines-2) - h.
            # Resume at next timestep: t_start = (num_lines-2) - h + 1 = num_lines - h - 1
            t_start = num_lines - max_horizon - 1
        
        pt_dict = utils.get_model_weights(
            base_model_dict,
            model_dir = args.model_dir,
            id_num = id_num)
        
        sim_params["id_num"] = id_num
        
        X = get_data(path)
        
        if args.debug:
            X = X[:100] #DEBUGGING ONLY
        
        targets = X[:,-1]
        
        t_end = len(X) - max_horizon
        
        if args.t_end is not None:
             t_end = min(args.t_end, t_end)

        if t_start >= t_end:
            continue
        
        sim_params["end"] = t_end
        
        os.makedirs(os.path.join(args.output_dir, "settings"), exist_ok = True)
        save_path = os.path.join(args.output_dir, f"settings/{id_num}.json")
        utils.save_sim_settings(sim_params | dir_params, save_path)
        
        sim_params["start"] = t_start
            
        models = get_model_dict(base_model_dict, pt_dict, args)
    
        forecasts = sim.get_online_forecasts(models, X, trainer, **sim_params)
        
        utils.save_data(forecasts, path = os.path.join(args.output_dir, "forecasts", f"{id_num}_forecasts.csv"))
        
        TimingRegistry.print_stats()
        
        if args.debug:
            break
    
    return None

def get_data(path):
    
    raw_data = pd.read_csv(path)
    
    X = data.transform_minute_data(raw_data)
    
    return X.reshape(-1,1)

   
def get_model_dict(base_model_dict, pt_dict, args):
        
    model_dict = dict()
    
    for model in base_model_dict.keys():
        weights = pt_dict.get(model)
        
        if weights is not None:
            
            base_model_dict[model].load_state_dict(
                torch.load(weights, weights_only = True),
                strict = False
            )
            
    model_dict["NODE"] = m.NODEForecaster(base_model_dict["NODE"])
    model_dict["HoltWinters"] = m.StatsForecaster(AutoETS, args.horizon)
    model_dict["ARIMA"] = m.StatsForecaster(AutoARIMA, args.horizon)
    model_dict["XGBoost"] = m.XGBoostForecaster(base_model_dict["XGBoost"], args.horizon)
    model_dict["NHITS"] = m.NNForecaster(base_model_dict["NHITS"])
    
    return model_dict


def get_args():
    
    parser = argparse.ArgumentParser(
        prog = "Test Forecast Horizons",
        description = "Testing Different Forecast Horizons for Minute Data"
    )
    
    for flag, settings in GlobalValues.command_line_args.items():
        
        parser.add_argument(flag, **settings)
    
    # #data access
    # parser.add_argument("--input_dir",type=str,default="")
    # parser.add_argument("--model_dir",type=str,default="./")
    # parser.add_argument("--output_dir",type=str,default="")
    # 
    # #training params
    # parser.add_argument("--epochs",type=int,default=50)
    # parser.add_argument("--batch",type=int,default=32)
    # parser.add_argument("--t_start",type=int,default=20)
    # parser.add_argument("--tolerance",type=int,default=100)
    # 
    # #simulation params (may also be training params)
    # parser.add_argument("--horizon",type=int,default=30)
    # parser.add_argument("--debug",type=bool,default=False)
    
    ## Add Arguments Specific to Script HERE
    
    args = parser.parse_args()
    return args


def raise_errors(args):
    
    if not args.input_dir:
        raise ValueError("Must Supply Valid Path to Input Data")
    if not args.output_dir:
        raise ValueError("Must Supply Valid Path to Output Data")


if __name__ == "__main__":
    
    main()
