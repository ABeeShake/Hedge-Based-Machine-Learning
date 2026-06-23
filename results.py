import numpy as np
import pandas as pd
import os
import csv
import argparse
import matplotlib.pyplot as plt
import matplotx
import matplotlib.patches as patches
import re

import ExpMethods.visualizations as viz
import ExpMethods.utils as utils

from glob import glob
from ExpMethods.globals import GlobalValues

def main():
    
    plt.rcParams.update(GlobalValues.plot_params)
    
    args = get_args()
    # print(args.omit_models)
    
    # print(args.bands)
    # print(type(args.bands))
    
    r_dir = os.path.join(args.output_dir, "regrets")
    l_dir = os.path.join(args.output_dir, "losses")
    i_dir = os.path.join(args.output_dir, "images")
    
    file_paths = glob(os.path.join(r_dir,f"*_fullregrets_eta{args.eta}.csv"))
    
    n_files = len(file_paths)
    shapes = [pd.read_csv(path).shape for path in file_paths]
    max_len = max([shape[0] for shape in shapes])
    max_col = max([shape[1] for shape in shapes])
    
    padded_arrays = []
    
    for path in file_paths:
        
        id_num = os.path.basename(path)[0:3]
        df = pd.read_csv(path)
        cols_to_drop = list(set(df.columns) & set(args.omit_models))
        df.drop(cols_to_drop,axis=1,inplace=True)
        arr = df.to_numpy()
        
        #print(id_num)
        
        padded_arrays.append(
            np.pad(
                arr, 
                (
                    (0,max_len - len(arr)), 
                    (0, max_col - arr.shape[1])
                    ),
                    mode = "constant",
                    constant_values = np.nan)
                    )
                    
        #print(padded_arrays[-1].shape)

    cols = df.columns.to_list()
    stacked_arr = np.stack(padded_arrays)
    
    avg_r_dict, q5_r_dict, q95_r_dict = get_plot_series(stacked_arr, cols)
    
    utils.save_data(
        avg_r_dict, 
        path = os.path.join(args.output_dir, "regrets",f"average_regrets_eta{args.eta}.csv")
        )
    utils.save_data(
        q5_r_dict, 
        path = os.path.join(args.output_dir, "regrets",f"q5_regrets_eta{args.eta}.csv")
        )
    utils.save_data(
        q95_r_dict, 
        path = os.path.join(args.output_dir, "regrets",f"q95_regrets_eta{args.eta}.csv")
        )
    
    omit = ["FS (Decay)", "FS (Uniform)", "FS (Decay2)","MoE"]
    if args.omit_models:
        omit = omit + args.omit_models
    
    if args.plot_all_regrets:
        plot_zoomed_regrets(
            avg_r_dict, 
            q5_r_dict, 
            q95_r_dict, 
            omit = omit, 
            start=100, 
            end=max_len-500,
            show = False,
            bands = args.bands,
            hedge_version = args.hedge_version,
            save_file = os.path.join(i_dir,"average_regrets_"+args.plot_file),
            title = args.plot_title)
    
    if args.plot_moe_regrets:
        plot_moe_regrets(
            avg_r_dict, 
            q5_r_dict, 
            q95_r_dict, 
            omit = [key for key in avg_r_dict.keys() if not key.startswith("Hedge")], 
            start=100, 
            end=max_len-500,
            show = False,
            bands = args.bands,
            save_file = os.path.join(i_dir,"average_regrets_MoE_"+args.plot_file),
            title = args.plot_title)
    
    if args.plot_bumpchart:    
        plot_bumpchart(
            avg_r_dict,
            omit = omit,
            start = 100,
            end = max_len - 500,
            show = False,
            hedge_version = args.hedge_version,
            save_file = os.path.join(i_dir,"expert_rankings_"+args.plot_file),
            title = args.plot_title)
    
    if args.plot_ceg:
        f_path = os.path.join(args.output_dir, "forecasts")
        plot_ceg(
            f_path,
            data_dir = args.data_dir,
            start = 100,
            end = -10,
            hedge_version = args.hedge_version,
            title = args.plot_title,
            show = False,
            save_file = os.path.join(i_dir,"clarke_grid_"+args.plot_file))
    
    return None


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
    parser.add_argument("--bands", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot_bumpchart", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot_title", type = str, default = "")
    parser.add_argument("--plot_file",type = str, default = "average_regrets.pdf")
    parser.add_argument("--omit_models", type = str, nargs = "*", default = "")
    parser.add_argument("--plot_ceg", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot_all_regrets", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot_moe_regrets", action=argparse.BooleanOptionalAction)
    parser.add_argument("--hedge_version", type = str, default = "Hedge (fd)")
    parser.add_argument("--eta", type = int, default = 10)
    
    args = parser.parse_args()
    return args


def plot_moe_regrets(avgrdict, q5rdict, q95rdict, omit = [], start=100, end=-100,**kwargs):
    
    title = kwargs.get("title","Average Regret Over Time Across All Patients")
    save_file = kwargs.get("save_file","average_regrets.pdf")
    show = kwargs.get("show",False)
    bands = kwargs.get("bands", True)
    
    avg_r_dict = {key: value for key,value in avgrdict.items() if key not in omit}
    q5_r_dict = {key: value for key,value in q5rdict.items() if key not in omit}
    q95_r_dict = {key: value for key,value in q95rdict.items() if key not in omit}
    
    plt.close("all")

    rep = {"(d)": "(decay)", "(fd)": "(fast decay)", "(c)": "(constant)"}
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
        
    for i,key in enumerate(avg_r_dict.keys()):
        T = (np.arange(len(avg_r_dict[key])) + 1)[start:end]
        Y = (avg_r_dict[key][start:end]/T)
        if key.startswith("Hedge"):
            label = pattern.sub(lambda m: rep[re.escape(m.group(0))], key)
            col = GlobalValues.color_params["Hedge"]
            ls = GlobalValues.linestyle_params[key]
        else:
            label = key
            col = GlobalValues.color_params[key]
            ls = "-"
        alpha = 1 if key.startswith("Hedge") else 0.7
        plt.plot(
            T, Y, 
            label = f"{label}", 
            color=col, 
            linestyle = ls,
            alpha = alpha,
            rasterized = True
            )
        if bands:
            plt.plot(T, (q5_r_dict[key][start:end]/T), 
            ls = ":", color=col, alpha = 0.2,
            rasterized = True)
            plt.plot(T, (q95_r_dict[key][start:end]/T), 
            ls = ":", color=col,alpha = 0.2,
            rasterized = True)
            plt.fill_between(T, 
            y1 = (q5_r_dict[key][start:end]/T), y2 = (q95_r_dict[key][start:end]/T), 
            color=col, alpha = 0.2,
            rasterized = True)
        
    plt.xlabel("Time Elapsed (min)")
    plt.ylabel("Regret Over Time ([mg/dL]/min)")
    plt.ylim(0,None)

    plt.title(title)
    plt.legend()
    
    if show:
        plt.show()
    if save_file:
        plt.savefig(save_file, transparent = True)
        
    return None



def plot_zoomed_regrets(avgrdict, q5rdict, q95rdict, omit = [], start=100, end=-100,**kwargs):
    
    title = kwargs.get("title","Average Regret Over Time Across All Patients")
    save_file = kwargs.get("save_file","average_regrets.pdf")
    show = kwargs.get("show",False)
    bands = kwargs.get("bands", True)
    hedge_version = kwargs.get("hedge_version",None)
    
    hedge_keys = list(filter(lambda x: re.findall("Hedge",string=x), avgrdict.keys()))
    
    if hedge_version:
        omit = omit + [key for key in hedge_keys if key != hedge_version]

    avg_r_dict = {key: value for key,value in avgrdict.items() if key not in omit}
    q5_r_dict = {key: value for key,value in q5rdict.items() if key not in omit}
    q95_r_dict = {key: value for key,value in q95rdict.items() if key not in omit}

    if hedge_version:
        avg_r_dict["Hedge"] = avg_r_dict.pop(hedge_version)
        q5_r_dict["Hedge"] = q5_r_dict.pop(hedge_version)
        q95_r_dict["Hedge"] = q95_r_dict.pop(hedge_version)
    
    plt.close("all")
    
    fig = plt.figure(figsize=(12,6))
    sub1 = fig.add_subplot(1,2,1)
    sub2 = fig.add_subplot(1,2,2)
    
    xmin, xmax, ymin, ymax = start, end, 0, 0
    
    rep = {"(d)": "(decay)", "(fd)": "(fast decay)", "(c)": "(constant)"}
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
        
    for i,key in enumerate(avg_r_dict.keys()):
        T = (np.arange(len(avg_r_dict[key])) + 1)[start:end]
        if xmax < 0:
            xmax = len(T)
        Y = (avg_r_dict[key][start:end]/T)
        
        if key.startswith("Hedge"):
            ymax = max(Y.max(),ymax)
            col = GlobalValues.color_params["Hedge"]
            ls = GlobalValues.linestyle_params[key]
            alpha = 1
        else:
            col = GlobalValues.color_params[key]
            ls = "-"
            alpha = 0.7
            
        sub1.plot(
            T, Y, 
            label = f"{key}: Average", 
            color=col, 
            linestyle = ls,
            alpha = alpha,
            rasterized = True
            )
        sub2.plot(
            T, Y, 
            label = f"{key}: Average", 
            color=col, 
            linestyle = ls,
            alpha = alpha,
            rasterized = True
            )
        if bands:
            sub1.plot(
                T, (q5_r_dict[key]/T)[start:end], 
                ls = ":", color=col, alpha = 0.2,
                rasterized = True)
            sub1.plot(
                T, (q95_r_dict[key]/T)[start:end], 
                ls = ":", color=col,alpha = 0.2,
                rasterized = True)
            sub1.fill_between(
                T, 
                y1 = (q5_r_dict[key]/T)[start:end], y2 = (q95_r_dict[key]/T)[start:end], 
                color=col, alpha = 0.2,
                rasterized = True)
        
    sub1.set_xlabel("Time Elapsed (min)")
    sub1.set_ylabel("Regret Over Time ([mg/dL]/min)")
    sub1.set_ylim(0,None)
    sub1.set_xlim(xmin,xmax)
    
    sub2.set_xlabel("Time Elapsed (min)")
    sub2.set_ylim(ymin,ymax+200)
    sub2.set_xlim(xmin,xmax)
    
    box = patches.Rectangle(
        (xmin,ymin),
        width = (xmax-xmin),
        height = (200 + ymax-ymin),
        linewidth = 10,
        linestyle = "--",
        edgecolor = "gray",
        facecolor = "none",
        rasterized = True)
    
    con1 = patches.ConnectionPatch(
        xyA = (xmax, ymin), coordsA = sub1.transData,
        xyB = (xmin,ymin), coordsB = sub2.transData,
        linestyle = "--"
        )
        
    con2 = patches.ConnectionPatch(
        xyA = (xmax, ymax+200), coordsA = sub1.transData,
        xyB = (xmin,ymax+200), coordsB = sub2.transData,
        linestyle = "--"
        )
        
    fig.add_artist(box)
    fig.add_artist(con1)
    fig.add_artist(con2)
        
        
    plt.suptitle(title)
    sub1.legend()
    
    if show:
        plt.show()
    if save_file:
        plt.savefig(save_file, transparent = True, dpi = 300)
        
    return None


def get_plot_series(stacked_arr, cols):

    avg_regrets = np.nanmean(stacked_arr, axis = 0)
    q5_regrets = np.nanquantile(stacked_arr, axis = 0, q = 0.05)
    q95_regrets = np.nanquantile(stacked_arr, axis = 0, q = 0.95)

    avg_r_dict = dict(zip(cols,avg_regrets.T))
    q5_r_dict = dict(zip(cols,q5_regrets.T))
    q95_r_dict = dict(zip(cols,q95_regrets.T))
    
    return avg_r_dict, q5_r_dict, q95_r_dict


def plot_bumpchart(avgrdict, omit = [], start = 100, end = -100, **kwargs):
    
    save_file = kwargs.get("save_file","average_regrets.pdf")
    title = kwargs.get("title")
    show = kwargs.get("show",False)
    hedge_version = kwargs.get("hedge_version","Hedge (fd)")
    
    hedge_keys = list(filter(lambda x: re.findall("Hedge",string=x), avgrdict.keys()))
    
    if hedge_version:
        omit = omit + [key for key in hedge_keys if key != hedge_version]
    
    avg_r_dict = {key: value for key,value in avgrdict.items() if key not in omit}
    
    if hedge_version:
        avg_r_dict["Hedge"] = avg_r_dict.pop(hedge_version)

    df = pd.DataFrame(avg_r_dict).drop("Hedge",axis=1).apply(lambda x: np.argsort(x).argsort()+1, axis = 1)
    df = df.iloc[start:end]

    df["Hedge"] = pd.DataFrame(avg_r_dict).apply(lambda x: np.argsort(x).argsort()+1, axis = 1).Hedge.iloc[start:end]
    
    plt.close("all")
    axes = viz.bumpchart(
        df, show_rank_axis= True,
        scatter = True, holes = True,
        line_args= {"linewidth": 2, "alpha": 0.8},
        scatter_args= {"s": 10, "alpha": 0.8},
        )
    axes[1].set_yticklabels([])
    axes[1].set_ylabel("")
    axes[1].tick_params(length = 0)
    axes[2].set_ylabel("Expert Ranking (1 = Highest Accuracy)")
    axes[2].yaxis.set_label_position("right")
    
    for tick_label in axes[0].get_yticklabels():
        if tick_label.get_text().startswith("Hedge"):
            tick_label.set_color(GlobalValues.color_params["Hedge"])
        else:
            tick_label.set_color(GlobalValues.color_params[tick_label.get_text()])
    
    rect = patches.Rectangle(
        (6.925, 0.75), 
        width = len(df)+200, height = .5, 
        linewidth = 1.5, linestyle = "--", 
        edgecolor = "red", facecolor = "none")
    axes[0].add_patch(rect)
    plt.text((6.925 + (len(df)+200)/2), 0.65, "Ideal Choice of Expert", ha = "center", va = "center", color = "red")
    plt.tight_layout()
    if title:
        plt.title(title)
    if show:
        plt.show()
    if save_file:
        plt.savefig(save_file, transparent = True, dpi = 300)


def plot_ceg(f_path, data_dir, start = 100, end = -100, **kwargs):

    save_file = kwargs.get("save_file")
    title = kwargs.get("title", "Clarke Error Grid")
    show = kwargs.get("show",False)
    hedge_version = kwargs.get("hedge_version","Hedge (fd)")
    
    main_dir=os.path.relpath(os.path.join(data_dir,os.pardir))

    actual = np.array([])
    predicted = np.array([])
    totals = np.array([])
    counts = pd.DataFrame(columns = ["id","A","B","C","D","E"])
    
    target_col = "Libre.GL" if os.path.basename(main_dir) == "weinstock" else "Dexcom.GL"
    
    for file in glob(f_path+"/*_fullforecasts.csv"):
        
        id_num = re.findall(r"\d{3}", file)[0]
        
        p = pd.read_csv(file)[hedge_version].iloc[start:end].to_numpy()
        a = pd.read_csv(data_dir+f"/{id_num}.csv")[target_col].iloc[start:end].to_numpy()
        
        predicted = np.concat([predicted,p])
        actual = np.concat([actual,a])
        counts.loc[len(counts),:] = dict(id = str(id_num)) | viz.clarke_error_grid(
            a,p, 
            return_dict = True, 
            save_file = None,
            plot = False)
    h,c,eta = re.findall(
        pattern = "h(.*)_c(.*)_eta(.*)\.", 
        string = os.path.basename(save_file))[0]
        
    print(f"Total CEG %s for h = {h}, c = {c}, eta = {eta}")
    viz.clarke_error_grid(
        actual,predicted, 
        print_pct = True, 
        **kwargs)
    
    counts["horizon"], counts["context"], counts["eta"] = h,c,eta
    counts["dataset"] = os.path.basename(main_dir)
        
    ceg_path = os.path.join(
        re.findall(pattern = "(.*)/h-.*", string = save_file)[0],
        "ceg.csv"
    )
    
    if not os.path.exists(ceg_path):
        counts.to_csv(ceg_path, mode = "w", header = True, index = False)
    else:
        counts.to_csv(ceg_path, mode = "a", header = False, index = False)

    pd.read_csv(ceg_path).drop_duplicates(
        subset = ["id","horizon","context","eta"],
        keep = "last").to_csv(ceg_path, index = False)

    return None


if __name__ == "__main__":
    
    main()
