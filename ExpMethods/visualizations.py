import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import *
from ExpMethods.utils import *
from ExpMethods.globals import GlobalValues

def plot_forecasts(forecasts:Dict[str, np.ndarray],targets: [torch.Tensor, np.ndarray], **kwargs):
    
    title = kwargs.get("title", "Model Forecasts")
    show = kwargs.get("show", False)
    save = kwargs.get("save",True)
    path = kwargs.get("path",None)
    
    targets = to_np(targets).flatten()
    
    t = np.arange(len(targets))
    
    plt.close("all")
    plt.figure()
    plt.plot(t, targets, label = "truth", c = "black")
    
    for model in forecasts.keys():
        
        f = forecasts[model]
        plt.plot(t, f, label = model)
        
    plt.legend()
    plt.title(title)
    plt.xlabel("Time Elapsed (min)")
    plt.ylabel("Blood Glucose (mg/dL)")
    
    if show:
        plt.show()
    if save:
        if path is None:
            raise ValueError("Must Supply Path to Save Image")
        plt.savefig(path)
    
    return None


def plot_losses(losses:Dict[str, np.ndarray], **kwargs):
    
    cumulative = kwargs.get( "cumulative", True)
    title = kwargs.get( "title", "Model Losses")
    show = kwargs.get("show", False)
    save = kwargs.get("save",True)
    path = kwargs.get("path",None)
    log_scale = kwargs.get("log_scale",False)
    
    t = np.arange(len(list(losses.values())[0]))
    
    plt.close("all")
    plt.figure()
    
    for model in losses.keys():
        
        l = losses[model]
        if cumulative:
            l = l.cumsum()
        plt.plot(t, l, label = model)
        
    plt.legend()
    plt.title(title)
    plt.xlabel("Time Elapsed (min)")
    plt.ylabel("Blood Glucose (mg/dL)")
    
    if log_scale:
        plt.yscale("log")
    
    if show:
        plt.show()
    if save:
        if path is None:
            raise ValueError("Must Supply Path to Save Image")
        plt.savefig(path)
    
    return None


def plot_regrets(regrets:Dict[str, np.ndarray], **kwargs):
    
    title = kwargs.get( "title", "Regrets")
    show = kwargs.get("show",False)
    save = kwargs.get("save",True)
    path = kwargs.get("path",None)
    over_t = kwargs.get("over_t", False)
    scale_func = kwargs.get("scale_func",lambda x: x)
    start = kwargs.get("start",0)
    end = kwargs.get("end", len(list(regrets.values())[0]))
    
    t = np.arange(len(list(regrets.values())[0])) + 1
    
    plt.close("all")
    plt.figure()
    if not over_t:
        plt.plot(t[start:end], t[start:end], label = "Linear Regret", c = "black", linewidth = 3, linestyle = "--")
    
    for method in regrets.keys():
        
        r = scale_func(regrets[method])
        if over_t:
            plt.plot(t[start:end], (r/t)[start:end], label = method)
        else:
            plt.plot(t[start:end], r[start:end], label = method)
        
    plt.legend()
    plt.title(title)
    plt.xlabel("Time Elapsed (min)")
    plt.ylabel("Regret (mg/dL)")
    if show:
        plt.show()
    if save:
        if path is None:
            raise ValueError("Must Supply Path to Save Image")
        plt.savefig(path)
    
    return None


def plot_aggregate_regrets(agg_regrets,**kwargs):
    
    title = kwargs.get("title", "Average Regret Over Time Across All Patients")
    show = kwargs.get("show",False)
    save = kwargs.get("save",True)
    path = kwargs.get("path",None)
    omit = kwargs.get("omit",[])
    start = kwargs.get("start",0)
    end = kwargs.get("end", agg_regrets["mean"].shape[0])

    T = np.arange(max_len) + 1
    
    plt.close("all")
    
    avg_r_dict, q5_r_dict, q95_r_dict = agg_regrets.values()
    
    for key in omit:
        del avg_r_dict[key]
        del q5_r_dict[key]
        del q95_r_dict[key]
    
    for i,key in enumerate(avg_r_dict.keys()):
        col = list(plt.cm.tab10(i))
        plt.plot(T[start:end], (avg_r_dict[key]/T)[start:end], label = f"{key}: Average", color=col)
        plt.plot(T[start:end], (q5_r_dict[key]/T)[start:end], ls = "--", color=col, alpha = 0.2)
        plt.plot(T[start:end], (q95_r_dict[key]/T)[start:end], ls = "--", color=col,alpha = 0.2)
        plt.fill_between(T[start:end], y1 = (q5_r_dict[key]/T)[start:end], y2 = (q95_r_dict[key]/T)[start:end], color=col, alpha = 0.2)
    
    plt.xlabel("Time Elapsed (min)")
    plt.ylabel("Regret Over Time ([mg/dL]/min)")
    plt.ylim(0,max([v.max() for v in agg_regrets.values()]))
    plt.title(title)
    plt.legend()
    if show:
        plt.show()
    if save:
        if path is None:
            raise ValueError("Must Supply Path to Save Image")
        plt.savefig(path)
        
        
def bumpchart(df, show_rank_axis= True, rank_axis_distance= 1.1, ax= None, scatter= False, holes= False,line_args= {}, scatter_args= {}, hole_args= {}):
    
    if ax is None:
        left_yaxis= plt.gca()
    else:
        left_yaxis = ax

    # Creating the right axis.
    right_yaxis = left_yaxis.twinx()
    
    axes = [left_yaxis, right_yaxis]
    
    # Creating the far right axis if show_rank_axis is True
    if show_rank_axis:
        far_right_yaxis = left_yaxis.twinx()
        axes.append(far_right_yaxis)
    
    for col in df.columns:
        y = df[col]
        x = df.index.values
        # Plotting blank points on the right axis/axes 
        # so that they line up with the left axis.
        for axis in axes[1:]:
            axis.plot(x, y, alpha= 0)

        left_yaxis.plot(x, y, **line_args, solid_capstyle='round')
        
        # Adding scatter plots
        if scatter:
            left_yaxis.scatter(x, y, **scatter_args)
            
            #Adding see-through holes
            if holes:
                bg_color = left_yaxis.get_facecolor()
                left_yaxis.scatter(x, y, color= bg_color, **hole_args)

    # Number of lines
    lines = len(df.columns)

    y_ticks = [*range(1, lines + 1)]
    
    # Configuring the axes so that they line up well.
    for axis in axes:
        axis.invert_yaxis()
        axis.set_yticks(y_ticks)
        axis.set_ylim((lines + 0.5, 0.5))
    
    # Sorting the labels to match the ranks.
    left_labels = df.iloc[0].sort_values().index
    right_labels = df.iloc[-1].sort_values().index
    
    left_yaxis.set_yticklabels(left_labels)
    right_yaxis.set_yticklabels(right_labels)
    
    # Setting the position of the far right axis so that it doesn't overlap with the right axis
    if show_rank_axis:
        far_right_yaxis.spines["right"].set_position(("axes", rank_axis_distance))
    
    return axes


def plot_weights(weights, **kwargs):
    
    names = kwargs.get("names", None)
    show = kwargs.get("show", False)
    save = kwargs.get("save",True)
    path = kwargs.get("path",None)

    row_sums = weights.sum(axis=1)
    weights = weights / row_sums[:, np.newaxis]

    T = weights.shape[0]
    t = np.arange(T)
    
    # Filter only to known colors if we want to be strict, or cycle
    colors = []
    for name in names:
        if name in GlobalValues.color_params:
            colors.append(GlobalValues.color_params[name])
        else:
            colors.append(None)  

    plt.close("all")
    fig, ax = plt.subplots()
    
    # Stackplot
    ax.stackplot(t, weights.T, labels=names, colors=colors, alpha=0.8)
    
    ax.set_xlim(0, T-1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time Step ($t$)', fontsize=14)
    ax.set_ylabel('Expert Weight ($w_{k,t}$)', fontsize=14)
    ax.set_title('Evolution of Expert Weights', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    
    # Legend outside
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=12)
    
    plt.tight_layout()
    
    if save:
        if path is None:
            raise ValueError("Must Supply Path to Save Image")
        plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show()


def clarke_error_grid(ref_values, pred_values, title="Clarke Error Grid", plot=True, show=False, save_file=None, return_dict=False, print_pct=False, **kwargs):
    
    #Checking to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(pred_values)), "Unequal Number of Observations"

    #Checks to see if the values are within the normal physiological range, otherwise it gives a warning
    # if max(ref_values) > 400 or max(pred_values) > 400:
    #     print("Input Warning: the maximum reference value is greater than 400 mg/dL.")
    # 
    # if min(ref_values) < 0 or min(pred_values) < 0:
    #     print("Input Warning: the minimum reference value is less than 0 mg/dL.")

    #Classification logic to separate points for coloring
    zones = {'A': ([], []), 'B': ([], []), 'C': ([], []), 'D1': ([], []), 'D2': ([], []), 'E1': ([], []), 'E2': ([], [])}
    counts = [0] * 7 # A, B, C, D1, D2, E1, E2

    for i in range(len(ref_values)):
        r = ref_values[i]
        p = pred_values[i]
        
        if (abs(r - p) <= 0.2*r) or (r < 70 and p < 70):
            zones['A'][0].append(r)
            zones['A'][1].append(p)
            counts[0] += 1
        elif (r <= 70 and p >= 180): # E1: hypo reference, hyper prediction
            zones['E1'][0].append(r)
            zones['E1'][1].append(p)
            counts[5] += 1
        elif (r >= 180 and p <= 70): # E2: hyper reference, hypo prediction
            zones['E2'][0].append(r)
            zones['E2'][1].append(p)
            counts[6] += 1
        elif ((r >= 70 and r <= 290) and p >= r + 110) or ((r >= 130 and r <= 180) and (p <= (7/5)*r - 182)):
            zones['C'][0].append(r)
            zones['C'][1].append(p)
            counts[2] += 1
        elif (r <= 70 and r > 0 and (p >= 70 and p <= 180)): # D1: hypo reference, normal prediction
            zones['D1'][0].append(r)
            zones['D1'][1].append(p)
            counts[3] += 1
        elif (r >= 240 and (p >= 70 and p <= 180)): # D2: hyper reference, normal prediction
            zones['D2'][0].append(r)
            zones['D2'][1].append(p)
            counts[4] += 1
        else:
            zones['B'][0].append(r)
            zones['B'][1].append(p)
            counts[1] += 1

    #Clear plot
    plt.close("all")

    #Set up plot
    if plot:
        fig, ax = plt.subplots(1, figsize=(8, 6))
        
        # Plot points by zone
        colors = {'A': 'green', 'B': 'yellow', 'C': 'orange', 'D1': 'red', 'D2': 'red', 'E1': 'darkred', 'E2': 'darkred'}
        for zone_char in ['A', 'B', 'C', 'D1', 'D2', 'E1', 'E2']:
            if len(zones[zone_char][0]) > 0:
                ax.scatter(zones[zone_char][0], zones[zone_char][1], marker='o', c=colors[zone_char], s=4, alpha=0.8, label=zone_char, rasterized = True)
        
        ax.set_title(title)
        ax.set_xlabel("Reference Concentration (mg/dL)")
        ax.set_ylabel("Prediction Concentration (mg/dL)")
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
        ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
        ax.set_xlim([0, 400])
        ax.set_ylim([0, 400])
        ax.set_aspect((400)/(400))
        
        #Plot zone lines
        ax.plot([0, 400], [0, 400], ':', c='black')                      #Theoretical perfect prediction
        ax.plot([0, 175/3], [70, 70], '-', c='black')
        ax.plot([175/3, 333.333], [70, 400], '-', c='black')
        ax.plot([70, 70], [84, 400], '-', c='black')                     #Zone D boundary
        ax.plot([0, 70], [180, 180], '-', c='black')                     #Zone E boundary
        ax.plot([70, 290], [180, 400], '-', c='black')                   #Zone A upper boundary
        ax.plot([70, 70], [0, 56], '-', c='black')
        ax.plot([70, 400], [56, 320], '-', c='black')
        ax.plot([180, 180], [0, 70], '-', c='black')                     #Zone E boundary
        ax.plot([180, 400], [70, 70], '-', c='black')                    #Zone D boundary
        ax.plot([240, 240], [70, 180], '-', c='black')                   #Zone D boundary
        ax.plot([240, 400], [180, 180], '-', c='black')                  #Zone D boundary
        ax.plot([130, 180], [0, 70], '-', c='black')                     #Zone C lower boundary
        
        # Add Zone labels
        ax.text(30, 15, "A", fontsize=15)
        ax.text(370, 260, "B", fontsize=15)
        ax.text(280, 370, "B", fontsize=15)
        ax.text(160, 370, "C", fontsize=15)
        ax.text(160, 15, "C", fontsize=15)
        ax.text(30, 140, "D1", fontsize=15)
        ax.text(370, 120, "D2", fontsize=15)
        ax.text(30, 370, "E1", fontsize=15)
        ax.text(370, 15, "E2", fontsize=15)


    total_pts = len(ref_values)
    
    if print_pct:
        print(f"Zone A: {counts[0]/total_pts*100:.2f}%")
        print(f"Zone B: {counts[1]/total_pts*100:.2f}%")
        print(f"Zone C: {counts[2]/total_pts*100:.2f}%")
        print(f"Zone D1: {counts[3]/total_pts*100:.2f}%")
        print(f"Zone D2: {counts[4]/total_pts*100:.2f}%")
        print(f"Zone E1: {counts[5]/total_pts*100:.2f}%")
        print(f"Zone E2: {counts[6]/total_pts*100:.2f}%")

    if save_file:
        plt.savefig(save_file, bbox_inches='tight')

    if show:
        plt.show()

    if return_dict:
        return {"A": counts[0], "B": counts[1], "C": counts[2], "D1": counts[3], "D2": counts[4], "E1": counts[5], "E2": counts[6]}

    return None
