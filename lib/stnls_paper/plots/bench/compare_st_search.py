
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
    print(results['errors_1'])
    print(results['errors_self_0'])
    print(results['errors_self_1'])
    etimes = results['exact_time'].to_numpy()
    dtimes = results['dtime'].to_numpy()
    print(etimes)
    labels = ["Ours" for _ in dtimes]
    labels += ["CPU" for _ in etimes]
    times = np.r_[dtimes,etimes]
    print(times)
    print(len(labels),len(times))
    df = pd.DataFrame({"Runtime":times,"Method":labels})
    df = df[::-1]
    print(df)

    # -- init plot --
    ginfo = {'width_ratios': [1.],'wspace':0.1, 'hspace':0.0,
             "top":0.9,"bottom":0.15,"left":0.19,"right":0.99}
    fig,ax = plt.subplots(1,1,figsize=(4,3),gridspec_kw=ginfo)

    # -- whiskers --
    sax = sns.barplot(x="Runtime", y="Method", ax=ax, data=df,
                      palette="muted", log=True)
    sax.bar_label(sax.containers[0], fmt='  %.03f')
    ax.set_xlim([0,15])


    # -- title --
    ax.set_title(r"Backpropagation Runtimes",fontsize=FSIZE)
    ax.set_ylabel("Method",fontsize=FSIZE)
    ax.set_xlabel("Runtime (seconds)",fontsize=FSIZE)

    # -- save figure --
    root = Path(str(SAVE_DIR))
    if not root.exists(): root.mkdir(parents=True)
    fn = str(root / "compare_st_search.png")
    print("Saving figure %s" % str(fn))
    plt.savefig(fn,dpi=500,transparent=True)
