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
import dnls
from dnls.utils.misc import rslice
from dnls.utils.timer import ExpTimer
from dnls.utils.inds import get_nums_hw
#get_batching_info

# -- local --
# import plots as plots

# -- plotting --
# from matplotlib import pyplot as plt
SAVE_DIR = Path("./output/race_cond/")

def compute_grad(search,noisy,ntotal,dists_grad):

    # -- copy noisy --
    noisy_g = noisy.clone()
    noisy_g.requires_grad_(True)

    # -- run forward --
    dists,inds = search(noisy_g[None,:],0,ntotal)
    dists,inds = dists[0],inds[0]
    th.cuda.synchronize()

    # -- start timer --
    timer = ExpTimer()
    timer.start("bwd")

    # -- run backward --
    th.autograd.backward(dists,dists_grad)

    # -- stop timer --
    th.cuda.synchronize()
    timer.stop("bwd")
    dtime = timer["bwd"]

    return noisy_g.grad,dtime

def expand_chnls(noisy,nchnls):
    nreps = int((nchnls-1)/noisy.shape[1]+1)
    noisy = repeat(noisy,'t c h w -> t (r c) h w',r=nreps)
    noisy = noisy[:,:nchnls]
    assert noisy.shape[1] == nchnls
    return noisy

def compute_exact_grad(search,noisy,ntotal,dists_grad,use_simp=False):

    # -- copy noisy --
    noisy_g = noisy.clone()
    if not use_simp:
        noisy_g.requires_grad_(True)

    # -- run forward --
    th.cuda.synchronize()
    dists,inds = search(noisy_g[None,:],0,ntotal)
    dists,inds = dists[0],inds[0]
    th.cuda.synchronize()

    # -- unpack vars --
    vid0,vid1 = noisy_g,noisy_g
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
        run_bwd = dnls.simple.search_bwd.run
        grad0,grad1 = run_bwd(dists_grad,vid0,vid1,inds,0,stride0,
                              ps,pt,dil,use_adj,reflect_bounds)
        grad = grad0+grad1
    else:
        th.autograd.backward(dists,dists_grad)
        grad = noisy_g.grad

    # -- stop timer --
    th.cuda.synchronize()
    timer.stop("bwd")
    dtime = timer["bwd"]

    return grad,dtime

def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)

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
    exact = cfg.exact# == "true"
    rbwd = cfg.rbwd# == "true"
    use_simp = cfg.use_simp# == "true"
    nbwd_mode = cfg.nbwd_mode
    print(exact,cfg.nbwd,rbwd,use_simp,nbwd_mode)

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

    # -- init search --
    esearch = dnls.search.init("l2_with_index",None,None,cfg.k,cfg.ps,
                               cfg.pt,cfg.ws,cfg.wt,-1,dil,stride0=cfg.stride0,
                               stride1=cfg.stride1,nbwd=1,use_adj=use_adj,
                               rbwd=False,exact=True)
    search = dnls.search.init("l2_with_index",None,None,cfg.k,cfg.ps,
                              cfg.pt,cfg.ws,cfg.wt,-1,dil,stride0=cfg.stride0,
                              stride1=cfg.stride1,nbwd=cfg.nbwd,use_adj=use_adj,
                              rbwd=rbwd,exact=exact,nbwd_mode=nbwd_mode,
                              ngroups=cfg.ngroups,npt=cfg.neigh_pt,qpt=cfg.query_pt)

    # -- batching info --
    t,c,h,w = noisy.shape
    nh,nw = get_nums_hw(noisy.shape,cfg.stride0,cfg.ps,dil,
                        pad_same=False,only_full=False)
    ntotal = t*nh*nw


    # -- forward and backward --
    emap = th.zeros_like(noisy)
    psnrs,errors = [],[]
    etime,dtime = 0,0
    for r in range(cfg.nreps):

        # -- new grad --
        grad = th.rand((ntotal,cfg.k),device=device)

        # -- compute exact grad --
        exact_grad,etime_i = compute_exact_grad(esearch,noisy,ntotal,grad,use_simp)
        print(etime_i)

        # -- compute proposed grad --
        vid_grad,dtime_i = compute_grad(search,noisy,ntotal,grad)
        print(dtime_i)

        # -- compute error --
        error = th.abs(vid_grad - exact_grad)/(exact_grad.abs() + 1e-5)
        emap += error/cfg.nreps
        error_m = th.mean(error).item()
        errors.append(error_m)

        # -- compute psnrs --
        imax = exact_grad.max()
        diff2 = (vid_grad/imax - exact_grad/imax)**2
        psnrs_i = -10 * th.log10(diff2.mean((1,2,3))).cpu().numpy()
        psnrs.append(psnrs_i)
        print("[%d] error_m: %2.6f" % (r,error_m))

        # -- save first example --
        if r == 0:
            c = exact_grad.shape[1]
            diff2 /= diff2.max().item()
            # print(error.mean().item())
            # print(error.max().item())
            # error /= (error.max().item()/2.)
            # error = th.clip(error,0.,1.)
            for ci in range(c):
                # print(diff2.shape,ci)
                fn = "diff_%s_%d" % (cfg.uuid,ci)
                dnls.testing.data.save_burst(diff2[:,[ci]],SAVE_DIR,fn)

        # -- accumuate deno --
        dtime += dtime_i
        etime += etime_i

    # -- save error map --
    # print(emap.max().item())
    # emap /= emap.max().item()
    for ci in range(c):
        print(ci,emap[:,ci].max().item())
        fn = "emap_nodiv_%s_%d" % (cfg.uuid,ci)
        dnls.testing.data.save_burst(emap[:,[ci]],SAVE_DIR,fn)

    emap /= emap.max().item()
    for ci in range(c):
        fn = "emap_%s_%d" % (cfg.uuid,ci)
        dnls.testing.data.save_burst(emap[:,[ci]],SAVE_DIR,fn)

    # -- average times --
    dtime /= cfg.nreps
    etime /= cfg.nreps

    # -- compute error --
    results = edict()
    results.errors = errors
    results.emap = emap.cpu().numpy()
    results.errors_m = np.mean(errors)
    results.errors_s = np.std(errors)
    results.psnrs = psnrs
    results.psnrs_m = np.mean(psnrs)
    results.dtime = dtime
    results.exact_time = etime

    return results

# def main():

#     # -- start info --
#     verbose = True
#     pid = os.getpid()
#     print("PID: ",pid)

#     # -- get cache --
#     cache_dir = ".cache_io"
#     cache_name = "race_cond" # most results here!
#     # cache_name = "race_cond_v2" # most results here!
#     cache = cache_io.ExpCache(cache_dir,cache_name)
#     # cache.clear()

#     # -- meshgrid --
#     exact = ["false"]
#     use_simp = ["false"]
#     # rbwd = ["true","false"]
#     # rbwd = ["true","false"]
#     rbwd = ["false"]#,"true"]
#     # nbwd_mode = ["median","mean"]
#     nbwd_mode = ["mean"]
#     nreps = [10]
#     # nbwd = [1,5,15,30]#,10,20]
#     # nchnls = [15]
#     nchnls = [3,15]
#     nbwd = [1,15] # this one.
#     # nchnls = [9]
#     # # nchnls = [1,2,3,5,10]
#     # nbwd = [1,100] # this one.
#     # nbwd = [1,5,7,9,11,15]#,10,20]
#     # nchnls = [3,6,9,15,30]

#     # nchnls = [30]
#     # nchnls = [3,9]
#     ws,wt,ps,k = [15],[3],[11],[30]
#     # ws,wt,ps,k = [8],[0],[3],[10]
#     # stride0,stride1 = [4],[1]
#     stride0,stride1 = [1],[1]
#     # isize = ["128_128"]
#     isize = ["156_156"]
#     # isize = ["256_256"]
#     # isize = ["512_512"]
#     # isize = ["32_32"]#,"256_256","512_512"]
#     # isize = ["96_96"]
#     # isize = ["64_64"]
#     exp_lists = {'exact':exact,'rbwd':rbwd,'nreps':nreps,
#                  "nchnls":nchnls,"ws":ws,"wt":wt,"ps":ps,"k":k,
#                  "stride0":stride0,"stride1":stride1,"isize":isize,
#                  "nbwd":nbwd,"use_simp":use_simp,"nbwd_mode":nbwd_mode}
#     exps = cache_io.mesh_pydicts(exp_lists) # create mesh

#     # -- striding impact --
#     exp_lists['nbwd'] = [15]
#     exp_lists['nchnls'] = [9]
#     exp_lists['stride0'] = [1,2,5]
#     exp_lists['stride1'] = [1,2,5]
#     # exps += cache_io.mesh_pydicts(exp_lists)

#     # -- search impact --
#     exp_lists['nbwd'] = [15]
#     exp_lists['nchnls'] = [9]
#     exp_lists['stride0'] = [1]
#     exp_lists['stride1'] = [1]
#     exp_lists['ws'] = [5,10,15,25]
#     exp_lists['wt'] = [1,3]
#     # exps += cache_io.mesh_pydicts(exp_lists)

#     # -- patchsize impact --
#     exp_lists['ps'] = [7,11,15]
#     exp_lists['k'] = [5,10,20,30]
#     # exps += cache_io.mesh_pydicts(exp_lists)

#     # -- settings --
#     cfg = edict()
#     cfg.dname = "set8"
#     cfg.vid_name = "sunflower"
#     # cfg.vid_name = "motorbike"
#     # cfg.vid_name = "hypersmooth"
#     cfg.sigma = 30.
#     cfg.flow = "true"
#     cfg.nframes = 3
#     cfg.frame_start = 0
#     cfg.frame_end = cfg.frame_start + cfg.nframes - 1
#     # cfg.isize = "156_156"
#     cfg.pt = 1
#     cfg.seed = 234
#     cache_io.append_configs(exps,cfg) # merge with default

#     # -- run configs --
#     nexps = len(exps)
#     for exp_num,exp in enumerate(exps):
#         # break
#         # continue

#         # -- info --
#         if verbose:
#             print("-="*25+"-")
#             print(f"Running experiment number {exp_num+1}/{nexps}")
#             print("-="*25+"-")
#             pp.pprint(exp)

#         # -- logic --
#         uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
#         cache.clear_exp(uuid)
#         results = cache.load_exp(exp) # possibly load result
#         if results is None: # check if no result
#             exp.uuid = uuid
#             results = run_exp(exp)
#             cache.save_exp(uuid,exp,results) # save to cache


#     #
#     # -- view --
#     #

#     clear_agg = True
#     save_agg = ".race_cond_agg.pkl"
#     records = cache.load_flat_records(exps,save_agg,clear_agg)
#     # print(records.columns)
#     fields = ['errors_m','dtime','exact_time','nchnls',
#               'nbwd','rbwd','nbwd_mode']
#     print(type(records))
#     print("-2.")
#     print(records[fields])
#     # print("-1.")
#     # print(records)
#     print("0.")
#     # plots.run(records)
#     exit(0)

#     # -- remove outliers --
#     errors = np.stack(records['errors'].to_numpy())
#     for m in range(errors.shape[0]):
#         quants_m = np.quantile(errors[m],[0.2,.8])
#         # print(quants_m)
#         bool_lb = errors[m] > quants_m[0]
#         bool_ub = errors[m] < quants_m[1]
#         bool_i = np.logical_and(bool_lb,bool_ub)
#         args = np.where(bool_i)
#         errors_f = errors[m][args]
#         print(m,np.mean(errors_f),np.std(errors_f),len(errors_f))


#     # -- save info --
#     save_dir = SAVE_DIR
#     if not save_dir.exists():
#         save_dir.mkdir(parents=True)

# if __name__ == "__main__":
#     main()

