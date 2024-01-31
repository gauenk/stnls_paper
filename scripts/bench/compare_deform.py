
"""

Compare with deformable convolution

"""

# -- imports --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- summary --
from torchinfo import summary as th_summary
from functools import partial
from easydict import EasyDict as edict
from dev_basics import net_chunks
import copy
dcopy = copy.deepcopy

# -- model io --
import importlib

# -- data --
import data_hub

# -- optical flow --
from dev_basics import flow

# -- bench --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt

# -- grid --
import stnls

# -- meta-learning --
from torchvision.ops import deform_conv2d,DeformConv2d

def load_sample(cfg):
    # -- init data --
    device = "cuda:0"
    imax = 255.
    data,loaders = data_hub.sets.load(cfg)
    dset = "tr" if not("dset" in cfg) else cfg['dset']
    if "vid_name" in cfg:
        indices = data_hub.filter_nframes(data[dset],cfg.vid_name,
                                          cfg.frame_start,cfg.nframes)
    else:
        indices = [0]
    sample = data[dset][indices[0]]
    vid = sample['noisy'][None,:].to(device)
    if "dd_in" in cfg and cfg.dd_in == 4:
        noise = th.zeros_like(vid[:,:,:1])
        vid = th.cat([vid,noise],-3)
    fflow = sample['fflow'][None,:].to(device)
    bflow = sample['bflow'][None,:].to(device)

    # -- expand channels --
    if "nchnls" in cfg:
        vid = repeat(vid[:,:,:1],'b t 1 h w -> b t r h w',r=cfg.nchnls)

    return vid,fflow,bflow

def run_deform_conv(cfg,vid,ngroups,ksize):
    vids = []
    B,T,C,H,W = vid.shape
    for ti in range(T):
        for tj in range(2*cfg.wt+1):
            # if ti == tj: continue
            offset = th.zeros(B,2*ngroups*ksize*ksize,H,W,device=vid.device)
            module = DeformConv2d(C,C,ksize,padding=(ksize//2,ksize//2)).to(vid.device)
            vid_ij = module(vid[:,ti],offset)
            vids.append(vid_ij)
    return [th.stack(vids)]

def run_nls_pair(cfg,vid,acc_flows):

    # -- search --
    cfg = dcopy(cfg)
    cfg.k = (cfg.k-1)//(2*cfg.wt+1)+1
    search = stnls.search.init(cfg)
    dists,inds = search.paired_vids(vid, vid, acc_flows, cfg.wt)
    stacking = stnls.tile.NonLocalStack(ps=cfg.ps,stride0=cfg.stride0,
                                        itype_fwd="float",itype_bwd="float")
    weights = th.exp(-dists)
    stack = stacking(vid,weights,inds)
    return [stack]

def run_nls(cfg,vid,fflow,bflow):
    search = stnls.search.init(cfg)
    dists,inds = search(vid,vid,fflow,bflow)
    stacking = stnls.tile.NonLocalStack(ps=cfg.ps,stride0=cfg.stride0,
                                        itype_fwd="float",itype_bwd="float")
    weights = th.exp(-dists)
    stack = stacking(vid,weights,inds)
    return [stack]

def run_method(cfg,name,vid,fflow,bflow,acc_flows,qinds):
    if name == "deform_conv":
        return run_deform_conv(cfg,vid,cfg.ngroups,cfg.ksize)
    elif name == "nls":
        return run_nls(cfg,vid,fflow,bflow)
    elif name == "paired":
        return run_nls_pair(cfg,vid,acc_flows)
    elif name == "refine":
        return run_refine(cfg,vid,fflow,bflow,qinds)
    else:
        raise ValueError(f"Uknown method name [{name}]")

def run_exp(cfg):

    # -- init --
    th.cuda.init()
    timer = ExpTimer()
    memer = GpuMemer()

    # -- read data --
    vid,fflow,bflow = load_sample(cfg)

    # -- acc flows --
    B,T,_,H,W = fflow.shape
    acc_flows = edict()
    acc_flows.fflow = th.zeros((B,T,T-1,2,H,W),device=fflow.device,dtype=fflow.dtype)
    acc_flows.bflow = th.zeros((B,T,T-1,2,H,W),device=fflow.device,dtype=fflow.dtype)


    # -- run search before for refine --
    dists,inds = None,None
    if cfg.search_name == "refine":
        dists,inds = run_nls(cfg,vid,fflow,bflow)

    # -- time forward --
    with th.no_grad():
        with TimeIt(timer,"fwd_nograd"):
            with MemIt(memer,"fwd_nograd"):
                run_method(cfg,cfg.search_name,vid,fflow,bflow,acc_flows,inds)

    # -- enable gradient --
    vid = vid.requires_grad_(True)

    # -- time forward --
    with TimeIt(timer,"fwd"):
        with MemIt(memer,"fwd"):
            outs = run_method(cfg,cfg.search_name,vid,fflow,bflow,acc_flows,inds)
    print("[%s] outs.shape: "%cfg.search_name,outs[0].shape)

    # -- time bwd --
    with TimeIt(timer,"bwd"):
        with MemIt(memer,"bwd"):
            loss = th.mean(outs[0])#th.mean([th.mean(o) for o in outs])
            loss.backward()

    # -- format results --
    results = edict()
    for key,val in timer.items():
        results[key] = val
    for key,(res,alloc) in memer.items():
        results["res_%s"%key] = res
        results["alloc_%s"%key] = alloc

    return results


def main():
    exp = edict()
    exp.dname = "set8"
    exp.vid_name = "sunflower"
    exp.dset = "te"
    exp.sigma = 50.
    exp.isize = "256_256"
    exp.read_flows = True
    exp.nframes = 5
    exp.frame_start = 0
    exp.frame_end = 4
    exp.ws = 9
    exp.wt = 1
    exp.dist_type = "l2"
    exp.itype_fwd = "float"
    exp.itype_bwd = "float"
    exp.stride1 = .2
    exp.ps = 1
    exp.nheads = 1
    exp.stride0 = 2
    exp.ksize = 3
    exp.ngroups = 12
    exp.nchnls = 108
    exp.k = 10
    exp.k_agg = 10
    exp.search_name = "deform_conv"

    # -- init --
    res0 = run_exp(exp)

    # -- deform --
    res0 = run_exp(exp)

    # -- our --
    exp.search_name = "nls"
    res1 = run_exp(exp)

    # -- our --
    exp.search_name = "paired"
    res2 = run_exp(exp)

    # -- our --
    # exp.search_name = "refine"
    # res3 = run_exp(exp)

    print(res0)
    print(res1)
    print(res2)
    # print(res3)


if __name__ == "__main__":
    main()
