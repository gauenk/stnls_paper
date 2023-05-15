"""
This graphic shows the errors incurred by the race condition

"""
# print("Run this in your springs computer.")

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
from stnls.utils.misc import rslice
from stnls.utils.timer import ExpTimer
from stnls.utils.inds import get_nums_hw
#get_batching_info

# -- utils --
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from dev_basics.utils.timer import ExpTimer,TimeIt

# -- local --
# import plots as plots
from .nat import NLSearch

# -- plotting --
# from matplotlib import pyplot as plt
SAVE_DIR = Path("./output/bench/")

def run_fwd(search,noisy):

    # -- copy noisy --
    noisy0 = noisy.clone()
    noisy1 = noisy.clone()
    noisy0.requires_grad_(True)
    noisy1.requires_grad_(True)

    # -- flows --
    T,F,H,W = noisy.shape
    fflow = th.zeros((1,T,2,H,W),device="cuda:0",dtype=th.float)
    bflow = th.zeros((1,T,2,H,W),device="cuda:0",dtype=th.float)

    # -- start timer --
    timer = ExpTimer()
    memer = GpuMemer()

    # -- warm-up --
    search(noisy0[None,:],noisy1[None,:],fflow,bflow)
    th.cuda.synchronize()

    # -- run forward --
    with MemIt(memer,"bwd"):
        with TimeIt(timer,"bwd"):
            dists,inds = search(noisy0[None,:],noisy1[None,:],fflow,bflow)
    th.cuda.synchronize()
    dists,inds = dists[0,0],inds[0]

    # -- stop timer --
    dtime = timer["bwd"]
    mem = memer['bwd']['res']

    return dtime,mem

def expand_chnls(noisy,nchnls):
    nreps = int((nchnls-1)/noisy.shape[1]+1)
    noisy = repeat(noisy,'t c h w -> t (r c) h w',r=nreps)
    noisy = noisy[:,:nchnls]
    assert noisy.shape[1] == nchnls
    return noisy

def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)

def rel_error(gt,est,tol=1e-5):
    error = th.abs(gt - est)/(gt.abs() + tol)
    return error

def init_from_cfg(cfg):
    return init(cfg)

def run_exp(cfg):

    # -- set seed --
    set_seed(cfg.seed)

    # -- data --
    index = 0
    data,loaders = data_hub.sets.load(cfg)
    groups = data.te.groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
    sample = data.te[indices[index]]

    # -- unpack config --
    device = "cuda:0"

    # -- unpack sample --
    region = sample['region']
    noisy = sample['noisy'].to(device)
    vid_frames = sample['fnums']
    dil = 1
    use_adj = False

    # -- append --
    noisy = expand_chnls(noisy,cfg.nchnls)

    # -- optional crop --
    noisy = rslice(noisy,region)
    print("[%d] noisy.shape: " % index,noisy.shape)

    # -- format --
    noisy /= 255.

    # -- patch search --
    _cfg = dcopy(cfg)
    _cfg.exact = False
    _cfg.use_adj = False
    _cfg.use_atomic = True
    _cfg.queries_per_thread = cfg.query_pt
    _cfg.neigh_per_thread = cfg.neigh_pt
    _cfg.channel_groups = cfg.ngroups
    if _cfg.search_name == "natten":
        nheads = 1
        search = NLSearch(_cfg.nchnls, nheads, k=_cfg.k, ps=_cfg.ws)
    else:
        search = stnls.search.init(_cfg)

    # -- forwward --
    time,mem = run_fwd(search,noisy)

    # -- results --
    results = edict()
    results.mem = mem
    results.time = time

    return results
