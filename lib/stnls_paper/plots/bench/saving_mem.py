
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
    print(results['mem'])
    print(results['errors_0'])
    
    mem = np.concatenate(results['mem'].to_numpy())
    labels = results['search_name'].to_numpy()
    labels = np.where(labels=="nls","ST-NLS",labels)
    labels = np.where(labels=="n3mm","ST-N3MM",labels)
    ps = results['ps'].to_numpy()
    stride0 = results['stride0'].to_numpy()
    grouping = ["(%d,%s)" % (_ps,_s0) for (_ps,_s0) in zip(ps,stride0)]
    df = pd.DataFrame({"Method":labels,"Memory":mem,"(Patchsize,Stride)":grouping})
    df = df[::-1]
    print(df)

    # -- init plot --
    ginfo = {'width_ratios': [1.],'wspace':0.1, 'hspace':0.0,
             "top":0.92,"bottom":0.15,"left":0.15,"right":0.97}
    fig,ax = plt.subplots(1,1,figsize=(5,3.4),gridspec_kw=ginfo)

    # -- whiskers --
    sax = sns.barplot(x="Memory", y="(Patchsize,Stride)", ax=ax, data=df,
                      palette="muted", log=True, hue="Method")
    sax.bar_label(sax.containers[0], fmt='  %.2f')
    sax.bar_label(sax.containers[1], fmt='  %.2f')
    ax.set_xlim([0,100])

    # -- title --
    ax.set_title(r"Memory Consumption",fontsize=FSIZE)
    ax.set_ylabel("(Patchsize, Stride)",fontsize=FSIZE)
    ax.set_xlabel("GPU Memory (GB)",fontsize=FSIZE)
    plt.legend(loc="lower right",title="Method")
    leg = ax.get_legend()
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_edgecolor((0, 0, 0, 1.))
    leg.get_frame().set_facecolor((0, 0, 0, 0.))

    # -- save figure --
    root = Path(str(SAVE_DIR))
    if not root.exists(): root.mkdir(parents=True)
    fn = str(root / "gpu_memory.png")
    print("Saving figure %s" % str(fn))
    plt.savefig(fn,dpi=500,transparent=True)


    plt.close("all")
    plt.clf()
