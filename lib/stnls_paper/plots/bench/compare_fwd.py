
"""

This figure compares the space-time search
with patches and without patches

"""


# -- misc --
import copy,os,random
dcopy = copy.deepcopy
import pprint
pp = pprint.PrettyPrinter(indent=4)

# -- data mng --
import pandas as pd

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- vision --
from torchvision.utils import make_grid,save_image
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as TF

# -- data io --
import data_hub

# -- caching --
import cache_io

# -- management --
from pathlib import Path
from easydict import EasyDict as edict

# -- results packages --
import stnls
from stnls.utils.misc import rslice,read_pickle,write_pickle
from stnls.utils.timer import ExpTimer
from stnls.utils.inds import get_nums_hw
#get_batching_info


# -- plotting --
import seaborn as sns
from matplotlib import lines as mlines
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# -- other --
FSIZE = 12
FSIZE_B = 14
FSIZE_S = 10
SAVE_DIR = Path("./output/plots/bench/")

def run(results):

    # -- extract --
    results = results[["mem","time","search_name","ps","ws","stride0","seed"]]
    rename = {"mem":"Memory","time":"Runtime","search_name":"name"}
    results = results.rename(columns=rename)
    results['Runtime'] *= 1000.
    results['name'].where(~results['name'].str.contains("n3mm"),
                           "ST-N3MM",inplace=True)
    results['name'].where(~results['name'].str.contains("nls"),
                           "ST-NLS",inplace=True)
    results['name'].where(~results['name'].str.contains("natten"),
                           "NATTEN",inplace=True)
    results['label'] = results['name'] + ", " + results['ps'].astype(str)
    print(results)
    print(results['label'].str.contains("natten"))
    results['label'].where(~results['label'].str.contains("NATTEN"),
                           "NATTEN",inplace=True)
    print(results)

    # -- viz --
    afields = ["Memory","Runtime"]
    gfields = ["name","ps","ws","stride0"]
    agg_fxn = lambda x: np.mean(np.stack(x))
    for f in afields: results[f] = results[f].apply(np.mean)
    summary = results.groupby(gfields).agg({k:agg_fxn for k in afields})
    summary = summary.reset_index()[gfields + afields]
    print(summary)


    # -- init plot --
    ginfo = {'width_ratios': [1.],'wspace':0.1, 'hspace':0.0,
             "top":0.92,"bottom":0.15,"left":0.13,"right":0.97}
    fig,ax = plt.subplots(1,1,figsize=(5,3.4),gridspec_kw=ginfo)

    # -- plot --
    # 
    # 

    fmt = {"ST-N3MM":"-+","ST-NLS":"-x","NATTEN":"-o"}
    cols = {"ST-N3MM":["#f05039","#e57a77"],
            "ST-NLS":["#1f449c","#7ca1cc"],
            "NATTEN":["k"]}
    for label,ldf in results.groupby("label"):
        x,y,yerr = [],[],[]
        for ws,wdf in ldf.groupby("ws"):
            x.append(ws)
            y.append(wdf['Runtime'].mean())
            yerr.append(wdf['Runtime'].std()/np.sqrt(len(wdf)))
        name = ldf['name'].iloc[0]
        ps = ldf['ps'].iloc[0]
        col_idx = 1 if ps == 7 else 0
        fmt_l = fmt[name]
        col_l = cols[name][col_idx]
        ax.errorbar(x,y,yerr=yerr,label=label,fmt=fmt_l,color=col_l)

    # -- title --
    ax.set_title(r"Forward Runtimes",fontsize=FSIZE)
    ax.set_ylabel("Rutime (milliseconds)",fontsize=FSIZE)
    ax.set_xlabel(r"Search Space $(S_s)$",fontsize=FSIZE)
    leg = ax.legend()
    leg = ax.get_legend()
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_edgecolor((0, 0, 0, 1.))
    leg.get_frame().set_facecolor((0, 0, 0, 0.))

    # -- save figure --
    root = Path(str(SAVE_DIR))
    if not root.exists(): root.mkdir(parents=True)
    fn = str(root / "compare_fwd.png")
    print("Saving figure %s" % str(fn))
    plt.savefig(fn,dpi=500,transparent=True)


    plt.close("all")
    plt.clf()
