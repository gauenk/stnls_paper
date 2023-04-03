"""

Plot to show the difference between our proposed search method
and the new search method

"""

# -- warnings --
import warnings

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- management --
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict
pd.options.mode.chained_assignment = None  # default='warn'

# -- interpolate --
from scipy.interpolate import LinearNDInterpolator,CloughTocher2DInterpolator
from scipy.interpolate import interpn,RegularGridInterpolator,RBFInterpolator
from scipy.interpolate import NearestNDInterpolator

# -- plotting --
from matplotlib import ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

# -- const --
SAVE_DIR = Path("./output/plots/compare_search/")

def run(records):
    plot_hist(records)

def plot_hist(records):

    # -- remove warnings --
    warnings.filterwarnings("ignore")


    # -- filter --
    records['mode'] = relabel_modes(records)
    records = records.rename(columns={"timer_fwd":"fwd_time","timer_bwd":"bwd_time"})
    records = records.rename(columns={"fwd_res":"fwd_mem","bwd_res":"bwd_mem"})
    records = average_aross_seed(records)
    records = records[["arch","mode","fwd_time","bwd_time","fwd_mem","bwd_mem"]]
    records = records.sort_values(by='mode')
    print(records[['arch','mode','bwd_time']])

    # -- arch names --
    arch = records['arch'].to_numpy()
    arch_l = np.array(["none" for _ in range(len(arch))])
    lnames = {"colanet":"COLA-Net","lidia":"LIDIA","n3net":"N3Net"}
    for key,val in lnames.items():
        arch_l = np.where(arch == key,val,arch_l)
    records['arch_l'] = arch_l

    # -- formatting numeric values --
    fields = ["fwd_time","bwd_time","fwd_mem","bwd_mem"]

    # -- plot constants --
    FSIZE = 12
    FSIZE_B = 14
    FSIZE_S = 10

    # -- get ticks --
    groups = edict()
    groups.time = ["fwd_time","bwd_time"]
    groups.mem = ["fwd_mem","bwd_mem"]
    ylims = edict()
    yticks = edict()
    yticklabels = edict()
    yticks_m = edict()
    for gname,group in groups.items():
        data = np.stack([records[g]['mean'].to_numpy() for g in group])
        data = np.log10(data)
        ymin,ymax = data.min(),data.max()
        ylims[gname] = [10**(ymin*1.)*.9,10**(ymax*1.)*1.1] # pos,pos
        yticks[gname] = [v for v in np.logspace(ymin,ymax,5)]
        yticklabels[gname] = ["%2.1e" % s for s in yticks[gname]]
        yticks_m[gname] = np.logspace(ymin,ymax,25)

    # -- tiles --
    titles = {"fwd_time":"Forward Time","bwd_time":"Backward Time",
              "fwd_mem":"Forward Memory","bwd_mem":"Backward Memory"}

    # -- init plot --
    ginfo = {'width_ratios': [.45,.45],'wspace':0.05, 'hspace':0.3,
             "top":0.95,"bottom":0.10,"left":0.15,"right":0.99}
    fig,axes = plt.subplots(2,2,figsize=(7,5),gridspec_kw=ginfo)

    # -- plot --
    axes = [axes[i][j] for i in range(2) for j in range(2)]
    for idx,field in enumerate(fields):

        # -- plot --
        df = records[['arch_l','mode',field]]
        df['mean'] = df[field]['mean']
        df['std'] = df[field]['std']
        df = df[['arch_l','mode','mean','std']]
        df[field] = df['mean']
        df_piv = df.pivot(index="arch_l",columns=["mode"],values=[field])
        df_std = df.pivot(index="arch_l",columns=["mode"],values=["std"])
        df_piv.plot(kind="bar",ax=axes[idx],rot=0,legend=False,
                    color=["red","blue","#FFC107"])#,yerr=df_std.to_numpy()/np.sqrt(5))
        # yerr looks gross.

        # -- format axis --
        group_name = "time" if "time" in field else "mem"
        axes[idx].set_title(titles[field],fontsize=FSIZE)
        axes[idx].set_yscale("log")
        blank = ["" for _ in yticks_m[group_name]]
        axes[idx].set_yticks(yticks_m[group_name],labels=blank,minor=True)
        if idx % 2 == 1:
            blank = ["" for _ in yticks[group_name]]
            axes[idx].set_yticks(yticks[group_name],labels=blank)
        else:
            axes[idx].set_yticks(yticks[group_name],yticklabels[group_name])
            if idx == 0:
                axes[idx].set_ylabel("Runtime (seconds)",fontsize=FSIZE)
            elif idx == 2:
                axes[idx].set_ylabel("Memory (GB)",fontsize=FSIZE)
        if idx >= 2:
            axes[idx].set_xlabel("Architecture")
        else:
            axes[idx].set_xlabel("")
        axes[idx].set_ylim(ylims[group_name])
        # axes[idx].get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    # -- legend --
    orig = mpatches.Patch([],color='red',label='Original')
    ours_0 = mpatches.Patch([],color='blue',label='Space')
    ours_3 = mpatches.Patch([],color='yellow',label='Space-\nTime')
    handles = [orig,ours_0,ours_3]
    leg = axes[0].legend(bbox_to_anchor=(0.00,1.), loc="upper left",
                        fontsize=FSIZE_S,title="Method",
                        title_fontsize=FSIZE_S,framealpha=1.,
                        edgecolor='k',ncol=1,handles=handles)
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((0, 0, 0, 0.0))

    # -- save figure --
    root = SAVE_DIR
    if not root.exists(): root.mkdir(parents=True)
    fn = str(root / "compare_search.png")
    print("Saving figure at %s" % fn)
    plt.savefig(fn,dpi=500,transparent=True)
    plt.close("all")

def average_aross_seed(records):
    gfields = ["arch","mode"]
    afields = ["fwd_time","bwd_time","fwd_mem","bwd_mem"]
    # print(records[gfields+afields])
    records = records.groupby(gfields).agg({f:["mean","std"] for f in afields})
    # records = records.groupby(gfields).agg({f:"mean"for f in afields})
    # print(records)
    records = records.reset_index()
    print(records)
    return records

def relabel_modes(records):
    _modes = records['mode'].to_numpy()
    _wt = records['wt'].to_numpy()
    modes = []
    for mode,wt in zip(_modes,_wt):
        if mode == "ours":
            modes.append(mode+" " + str(wt))
        else:
            modes.append(mode)
    return modes

