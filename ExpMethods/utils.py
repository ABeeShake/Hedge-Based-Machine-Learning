"""ExpMethods/utils.py — Shared utility functions for the HBML pipeline.

Provides I/O helpers, array conversion utilities, and result aggregation
routines used across ``run_simulation.py``, ``process_results.py``, and the
post-hoc analysis scripts.

Key functions
-------------
* ``to_np``               — converts a torch Tensor or ndarray to a NumPy array.
* ``save_data``           — writes a dict of arrays to a CSV file.
* ``make_matrix``         — stacks a dict of 1-D arrays into a T × N matrix.
* ``get_model_weights``   — locates the most recent saved checkpoint for each
  torch-based expert given a patient ID and model directory.
* ``load_results_from_csv`` / ``load_targets_from_csv`` — reload forecast
  or target arrays from CSV for post-hoc processing.
* ``save_sim_settings`` / ``load_sim_settings`` — serialize/deserialize the
  simulation parameter dict as JSON for reproducible runs.
* ``stack_results``       — pad per-patient result arrays to the same length
  and stack by diabetes subgroup (T2D, pre-T2D, non-T2D).
* ``aggregate_results``   — compute mean and 5th/95th percentile across
  patients for a stacked result array.
"""
import os

import numpy as np
import pandas as pd
import torch
import json
import re

from ExpMethods.globals import GlobalValues
from glob import glob


def to_np(array: [torch.Tensor,np.ndarray]):
    
    return array if isinstance(array, np.ndarray) else array.detach().cpu().numpy()


def save_data(collection, **kwargs):
    
    path = kwargs.get("path", None)
    mode = kwargs.get("mode", "w")
    header = kwargs.get("header", True)
    
    pd.DataFrame(collection).to_csv(path, index = False, mode = mode, header = header)
    
    return None


def make_matrix(collection):
    
    return np.stack(tuple(collection.values()), axis = 1)


def get_model_weights(model_dict,**kwargs):
    
    model_dir = kwargs.get("model_dir", "./")
    id_num = kwargs.get("id_num","000")
    
    pt_dict = dict()
    
    model_sort = lambda x: int(re.findall("[0-9]+_[a-z]+_iteration([0-9]+)",x)[0])
    
    for model in filter(lambda m: m.casefold() in GlobalValues.torch_models, model_dict.keys()):
        
        model_name = model.casefold()
        #print(model_name)
        
        dir = os.path.join(model_dir,f"{model_name}")
        
        dir_exists = os.path.exists(dir)
        pretrained_model = glob(os.path.join(dir,f"pretrained_{model_name}.pt"))
        current_models = glob(os.path.join(dir,f"{id_num}*.pt"))
        
        #print(not dir_exists or (not current_models and not pretrained_model))
        #print(not current_models and pretrained_model)
        
        if not dir_exists or (not current_models and not pretrained_model):
            pt_dict[model] = None
        elif not current_models and pretrained_model:
            pt_dict[model] = pretrained_model[0]
        else:
            pt_dict[model] = sorted(current_models,key = model_sort)[-1]
        
    return pt_dict


def load_results_from_csv(path, **kwargs):
    
    df = pd.read_csv(path, **kwargs)
    return {col: df[col].to_numpy() for col in df.columns}


def load_targets_from_csv(path, **kwargs):
    
    df = pd.read_csv(path, **kwargs)
    return df["Libre.GL" if "Libre.GL" in df.columns else "Dexcom.GL"].to_numpy()


def save_sim_settings(setting_dict, save_path):
    with open(save_path, "w+") as fp:
        json.dump(setting_dict, fp)


def load_sim_settings(path):
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


def get_processed_files(input_dir,output_dir):
    
    filepaths = os.path.join(output_dir, "forecasts","*_forecasts.csv")
    
    input_files = [
        os.path.join(input_dir,f"CGMacros-{os.path.basename(file)[0:3]}-clean.csv") 
        for file 
        in glob(filepaths)
        ]
    
    return input_files


def stack_results(paths, data_classes = GlobalValues.CGMacros):
    
    max_len = max([len(pd.read_csv(path)) for path in paths])
    
    diabetic_padded = []
    nondiabetic_padded = []
    prediabetic_padded = []
    
    
    for path in paths:
        
        id_num = os.path.basename(path)[0:3]
        
        arr = pd.read_csv(path).to_numpy()
        
        if id_num in data_classes["diabetic_ids"]:
            diabetic_padded.append(
                np.pad(
                    arr, 
                    (
                        (0,max_len - len(arr)), 
                        (0, 0)
                        ),
                    mode = "constant",
                    constant_values = np.nan)
                        )
        elif id_num in data_classes["nondiabetic_ids"]:
            nondiabetic_padded.append(
                np.pad(
                    arr, 
                    (
                        (0,max_len - len(arr)), 
                        (0, 0)
                        ),
                    mode = "constant",
                    constant_values = np.nan)
                        )
        else:
            prediabetic_padded.append(
                np.pad(
                    arr, 
                    (
                        (0,max_len - len(arr)), 
                        (0, 0)
                        ),
                    mode = "constant",
                    constant_values = np.nan)
                        )
        
        
        diabetic_stacked = np.stack(diabetic_padded)
        nondiabetic_stacked = np.stack(nondiabetic_padded)
        prediabetic_stacked = np.stack(prediabetic_padded)
        full_stacked = np.stack(diabetic_padded+nondiabetic_padded+prediabetic_padded)
        
        return dict(
        diabetic = diabetic_stacked,
        nondiabetic = nondiabetic_stacked,
        prediabetic = prediabetic_stacked,
        combined = full_stacked)


def aggregate_results(stacked_arr, cols):
    
    avg_regrets = np.nanmean(stacked_arr, axis = 0)
    q5_regrets = np.nanquantile(stacked_arr, axis = 0, q = 0.05)
    q95_regrets = np.nanquantile(stacked_arr, axis = 0, q = 0.95)

    avg_r_dict = dict(zip(cols,avg_regrets.T))
    q5_r_dict = dict(zip(cols,q5_regrets.T))
    q95_r_dict = dict(zip(cols,q95_regrets.T))
    
    return dict(
    mean = avg_r_dict,
    q5 = q5_r_dict, 
    q95 = q95_r_dict)
