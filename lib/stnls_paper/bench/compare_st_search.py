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

# -- plotting --
# from matplotlib import pyplot as plt
SAVE_DIR = Path("./output/bench/")

def compute_grad(search,noisy,dists_grad,cfg):

    # -- copy noisy --
    noisy0 = noisy.clone()
    noisy1 = noisy.clone()
    noisy0.requires_grad_(True)
    noisy1.requires_grad_(True)

    # -- start timer --
    timer = ExpTimer()
    memer = GpuMemer()

    # -- run forward --
    T,F,H,W = noisy.shape
    fflow = th.zeros((1,T,2,H,W),device="cuda:0",dtype=th.float)
    bflow = th.zeros((1,T,2,H,W),device="cuda:0",dtype=th.float)
    with MemIt(memer,"bwd"):
        if cfg.search_name == "nls":
            bs = -1 #cfg.batchsize
            dists,inds = search(noisy0[None,:],noisy1[None,:],fflow,bflow,bs)
        else:
            dists,inds = search(noisy0[None,:],noisy1[None,:],fflow,bflow)
    dists,inds = dists[0,0],inds[0]
    th.cuda.synchronize()

    # -- run backward --
    with TimeIt(timer,"bwd"):
        th.autograd.backward(dists,dists_grad)

    # -- stop timer --
    th.cuda.synchronize()
    dtime = timer["bwd"]
    mem = memer['bwd']['res']
    return noisy0.grad,noisy1.grad,dtime,mem

def expand_chnls(noisy,nchnls):
    nreps = int((nchnls-1)/noisy.shape[1]+1)
    noisy = repeat(noisy,'t c h w -> t (r c) h w',r=nreps)
    noisy = noisy[:,:nchnls]
    assert noisy.shape[1] == nchnls
    return noisy

def compute_exact_grad(search,noisy,ntotal,dists_grad,dist_type,use_simp=False):

    # -- copy noisy --
    # noisy_g = noisy.clone()
    noisy0 = noisy.clone()
    noisy1 = noisy.clone()
    if not use_simp:
        noisy0.requires_grad_(True)
        noisy1.requires_grad_(True)

    # -- run forward --
    th.cuda.synchronize()
    T,F,H,W = noisy.shape
    fflow = th.zeros((1,T,2,H,W),device="cuda:0",dtype=th.float)
    bflow = th.zeros((1,T,2,H,W),device="cuda:0",dtype=th.float)
    dists,inds = search(noisy0[None,:],noisy1[None,:],fflow,bflow)
    dists,inds = dists[0,0],inds[0,0]
    th.cuda.synchronize()

    # -- unpack vars --
    stride0 = search.stride0
    ps,pt = search.ps,search.pt
    dil = search.dilation
    use_adj = search.use_adj
    reflect_bounds = search.reflect_bounds

    # -- start timer --
    timer = ExpTimer()
    timer.start("bwd")

    # -- exec --
    if use_simp:
        run_bwd = stnls.simple.search_bwd.run
        grad0,grad1 = run_bwd(dists_grad,noisy0,noisy1,inds,0,stride0,
                              ps,pt,dil,use_adj,reflect_bounds,dist_type=dist_type)
    else:
        grad0 = th.zeros_like(noisy0)
        grad1 = th.zeros_like(noisy0)

    # -- stop timer --
    th.cuda.synchronize()
    timer.stop("bwd")
    dtime = timer["bwd"]

    return grad0,grad1,dtime

def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)

def rel_error(gt,est,tol=1e-5):
    error = th.abs(gt - est)/(gt.abs() + tol)
    return error

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
    use_simp = cfg.use_simp# == "true"

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
    search = stnls.search.init(_cfg)

    # -- batching info --
    t,c,h,w = noisy.shape
    nh,nw = get_nums_hw(noisy.shape,cfg.stride0,cfg.ps,dil,
                        pad_same=False,only_full=False)
    ntotal = t*nh*nw


    # -- forward and backward --
    emap0,emap1 = th.zeros_like(noisy),th.zeros_like(noisy)
    emaps = [emap0,emap1]
    psnrs = [[],[]]
    errors = [[],[]]
    errors_self = [[],[]]
    mem = []
    etime,dtime = 0,0
    for r in range(cfg.nreps):

        # -- new grad --
        # grad = th.rand((ntotal,cfg.k),device=device)
        grad = th.ones((ntotal,cfg.k),device=device)

        # -- compute exact grad --
        egrad0,egrad1,etime_i = compute_exact_grad(search,noisy,ntotal,grad,
                                                   cfg.dist_type,use_simp)
        # print("-"*3)
        # print("etime: ",etime_i)
        # print(egrad0[0,0,:10,:10])

        # -- compute proposed grad --
        grad0,grad1,dtime_i,mem_i = compute_grad(search,noisy,grad,cfg)
        # print(grad0.shape)
        # print(grad0[0,0,:10,:10])
        # print("dtime: ",dtime_i)

        # # -- 2nd time --
        rgrad0,rgrad1,dtime_i2,mem_i2 = compute_grad(search,noisy,grad,cfg)
        # print(rgrad0[0,0,:10,:10])
        print("etime,dtime: ",etime_i,dtime_i)
        # print("diff0: ",th.mean((rgrad0 - grad0)**2).item())
        # print("diff1: ",th.mean((rgrad1 - grad1)**2).item())

        # -- memory --
        mem.append(mem_i)

        # -- self errors --
        errors_self[0].append(th.mean(rel_error(grad0,rgrad0)).item())
        errors_self[1].append(th.mean(rel_error(grad1,rgrad1)).item())

        # -- each grad --
        grads = [grad0,grad1]
        egrads = [egrad0,egrad1]
        for i in range(2):
            # -- compute error --
            error = th.abs(grads[i] - egrads[i])/(egrads[i].abs() + 1e-5)
            emaps[i] += error/cfg.nreps
            error_m = th.mean(error).item()
            errors[i].append(error_m)
    
            # -- compute psnrs --
            imax = egrads[i].max()
            diff2 = (grads[i]/imax - egrads[i]/imax)**2
            psnrs_i = -10 * th.log10(diff2.mean((1,2,3))).cpu().numpy()
            psnrs[i].append(psnrs_i)
            print("[%d] error_m: %2.6f" % (r,error_m))

        # -- save first example --
        if r == 0:
            c = egrad0.shape[1]
            diff2 = (grads[0]/imax - egrads[0]/imax)**2
            diff2 /= diff2.max().item()
            # print(error.mean().item())
            # print(error.max().item())
            # error /= (error.max().item()/2.)
            # error = th.clip(error,0.,1.)
            for ci in range(c):
                # print(diff2.shape,ci)
                fn = "diff_%s_%d" % (cfg.uuid,ci)
                # stnls.testing.data.save_burst(diff2[:,[ci]],SAVE_DIR,fn)

        # -- accumuate deno --
        dtime += dtime_i
        etime += etime_i

    # -- save error map --
    # print(emap.max().item())
    # emap /= emap.max().item()
    for i in range(2):
        for ci in range(c):
            print(ci,emaps[i][:,ci].max().item())
            fn = "emap%d_nodiv_%s_%d" % (i,cfg.uuid,ci)
            # stnls.testing.data.save_burst(emaps[i][:,[ci]],SAVE_DIR,fn)
    
        emaps[i] /= emaps[i].max().item()
        for ci in range(c):
            fn = "emap%d_%s_%d" % (i,cfg.uuid,ci)
            # stnls.testing.data.save_burst(emaps[i][:,[ci]],SAVE_DIR,fn)

    # -- average times --
    dtime /= cfg.nreps
    etime /= cfg.nreps

    # -- compute error --
    results = edict()
    results["mem"] = mem
    for i in range(2):
        results["errors_%d"%i] = errors[i]
        results["errors_self_%d"%i] = errors_self[i]
        results["emap_%d"%i] = emaps[i].cpu().numpy()
        results["errors_m_%d"%i] = np.mean(errors[i])
        results["errors_s_%d"%i] = np.std(errors[i])
        results["psnrs_%d"%i] = psnrs[i]
        results["psnrs_m_%d"%i] = np.mean(psnrs[i])
    results.dtime = dtime
    results.exact_time = etime

    return results
