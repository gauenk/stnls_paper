"""

Compare aggregation

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
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

# -- const --
SAVE_DIR = Path("./output/plots/compare_agg/")

def run(records):
    create_hist(records)

def create_hist(records):

    # -- remove warnings --
    warnings.filterwarnings("ignore")

    # -- filter --
    records = records.rename(columns={"timer_fwd":"fwd_time","timer_bwd":"bwd_time"})
    records = records.rename(columns={"fwd_res":"fwd_mem","bwd_res":"bwd_mem"})
    records = records[["arch","mode","fwd_time","bwd_time","fwd_mem","bwd_mem"]]
    records = records.sort_values(by='mode')
    fields = ["fwd_time","bwd_time","fwd_mem","bwd_mem"]
    for field in fields:
        records[field] = records[field].apply(lambda x:x[0])

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
        data = np.concatenate(records[group].to_numpy())
        data = data.ravel()
        data = np.log10(data)
        ymin,ymax = data.min(),data.max()
        ylims[gname] = [10**(ymin*1.)*.9,10**(ymax*1.)*1.1] # pos,pos
        yticks[gname] = [v for v in np.logspace(ymin,ymax,5)]
        yticklabels[gname] = ["%2.1e" % s for s in yticks[gname]]
        yticks_m[gname] = np.logspace(ymin,ymax,25)

    # -- tiles --
    titles = {"time":"Runtime","mem":"Memory"}

    # -- init plot --
    ginfo = {'width_ratios': [.28,.28],'wspace':0.45,
             "top":0.88,"bottom":0.12,"left":0.15,"right":0.99}
    fig,axes = plt.subplots(1,2,figsize=(6,2),gridspec_kw=ginfo)
    axes = [axes[i] for i in range(2)]

    # -- plot --
    gkeys = sorted(list(groups.keys()))
    for idx,group in enumerate(gkeys):

        # -- plot --
        fields = groups[group]
        rname = {v:"Forward" if "fwd" in v else "Backward" for v in fields}
        df = records[['mode'] + fields]
        df = df.set_index("mode")
        df = df.rename(columns=rname).T
        print(df)
        df.plot(kind="bar",ax=axes[idx],rot=0,legend=False,color=["b","orange"])

        # -- format axis --
        axes[idx].set_title(titles[group],fontsize=FSIZE)
        axes[idx].set_yscale("log")
        blank = ["" for _ in yticks_m[group]]
        axes[idx].set_yticks(yticks_m[group],labels=blank,minor=True)
        if False:#idx % 2 == 1:
            blank = ["" for _ in yticks[group]]
            axes[idx].set_yticks(yticks[group],labels=blank)
        else:
            axes[idx].set_yticks(yticks[group],yticklabels[group])
            if idx == 0:
                axes[idx].set_ylabel("Runtime (seconds)",fontsize=FSIZE)
            elif idx == 1:
                axes[idx].set_ylabel("Memory (GB)",fontsize=FSIZE)
        axes[idx].set_xlabel("")
        axes[idx].set_ylim(ylims[group])

    # -- legend --
    blue = mpatches.Patch([],color='blue',label='Original')
    orange = mpatches.Patch([],color='orange',label='Ours')
    handles = [blue,orange]
    leg = axes[0].legend(bbox_to_anchor=(-0.02,1.03), loc="upper left",
                        fontsize=FSIZE_S,title="Method",
                        title_fontsize=FSIZE_S,framealpha=1.,
                        edgecolor='k',ncol=1,handles=handles)
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((0, 0, 0, 0.0))

    # -- save figure --
    root = SAVE_DIR
    if not root.exists(): root.mkdir(parents=True)
    fn = str(root / "compare_agg.png")
    print("Saving figure at %s" % fn)
    plt.savefig(fn,dpi=500,transparent=True)
    plt.close("all")


