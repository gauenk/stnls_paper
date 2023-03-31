
# -- exp --
import os
import torch as th
import pandas as pd
from easydict import EasyDict as edict

import torch.nn.functional as F
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt

# -- local --
from .api import init_agg
from ..search import init_search

def run(cfg):
    # chnls = [8]
    # archs = ["n3net"]
    # modes = ["original","ours"]

    # for C,arch in zip(chnls,archs):

    # -- init results --
    results = edict({"arch":[],"mode":[]})

    # -- init search --
    C = cfg.chnls
    arch = cfg.arch
    mode = cfg.mode
    search = init_search(cfg,arch,'ours') # same for all

    # -- alloc --
    vid = th.rand((1,cfg.T,C,cfg.H,cfg.W),device="cuda:0")
    # vid.requires_grad_(True)
    dists,inds = search(vid)
    # vid = vid[0]
    # dists,inds = dists[0],inds[0] # no heads
    print("min,max: ",inds.min(),inds.max())
    scale = 10
    dists = F.softmax(scale*-dists,-1)

    # -- init agg --
    sfxn = init_agg(cfg,arch)
    print(sfxn['original'].stride0)
    print(sfxn['ours'].stride0)

    # for mode in modes:

    # -- init --
    timer = ExpTimer()
    memer = GpuMemer()

    # -- burn-in --
    _ = sfxn[mode](vid,dists,inds)
    th.cuda.synchronize()

    # -- with backward --
    dists.requires_grad_(True)

    # -- forward --
    name = "fwd"
    with TimeIt(timer,name):
        with MemIt(memer,name):
            patches = sfxn[mode](vid,dists,inds)

    # -- backward --
    name = "bwd"
    patches_grad = th.randn_like(patches)
    with TimeIt(timer,name):
        with MemIt(memer,name):
            th.autograd.backward(patches,patches_grad)

    # -- format times --
    results.arch.append(arch)
    results.mode.append(mode)
    for key,val in timer.items():
        if key in results:
            results[key].append(val)
        else:
            results[key] = [val]
    for key,(res,alloc) in memer.items():
        for f,mem in zip(["res","alloc"],[res,alloc]):
            res_key = "%s_%s" % (key,f)
            if res_key in results:
                results[res_key].append(res)
            else:
                results[res_key] = [res]

    print(results)
    return results
