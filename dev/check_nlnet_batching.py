"""

   I see strange difference in validation across (i2Lab+Springs) & Gilbreth.
   -> Maybe difference PCs have different random seeds.
   -> Maybe differences in image crops from "davis_cropped"
   -> Maybe differences in .... batch size???

   This script shows our issue is not batch size...

"""

# -- basic --
import torch as th
import numpy as np
from einops import rearrange,repeat
from pathlib import Path
from easydict import EasyDict as edict
from dev_basics.utils import vid_io

# -- vid/img io --
from dev_basics.utils import vid_io

# -- exps --
from dev_basics.utils.misc import set_seed

# -- optical flow --
from dev_basics import flow

# -- data --
import data_hub

# -- non-local opts --
import stnls

# -- network in question --
import nlnet

def get_grads(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.ravel())
    grads = th.cat(grads)
    return grads

def zero_grads(model):
    for param in model.parameters():
        param.grad[...] = 0

def get_video(cfg):
    device = "cuda:0"
    data,loaders = data_hub.sets.load(cfg)
    get_inds = lambda name: data_hub.filter_subseq(data[cfg.dset],name,0,cfg.nframes)[0]
    indices = [get_inds(vname) for vname in cfg.vid_names]
    noisy,clean = [],[]
    for i in indices:
        noisy.append(data[cfg.dset][i]['noisy'].to(device)/255.)
        clean.append(data[cfg.dset][i]['clean'].to(device)/255.)
    noisy = th.stack(noisy)
    clean = th.stack(clean)
    return noisy,clean

def run(cfg):

    # -- init --
    set_seed(cfg.seed)

    # -- load sample --
    noisy,clean = get_video(cfg)
    print("noisy.shape: ",noisy.shape)
    print("clean.shape: ",clean.shape)
    B,T,F,H,W = noisy.shape

    # -- load model --
    model = nlnet.load_model(cfg).to(cfg.device)

    # -- fwd each --
    deno0 = []
    for bi in range(B):
        deno0.append(model(noisy[[bi]]))
    deno0 = th.cat(deno0)
    loss0 = th.mean((deno0-clean)**2)
    loss0.backward()
    grads0 = get_grads(model)

    # -- reset --
    zero_grads(model)

    # -- fwd batch --
    deno1 = model(noisy)
    loss1 = th.mean((deno1-clean)**2)
    loss1.backward()
    grads1 = get_grads(model)

    # -- compare --
    print("denos: ",th.mean((deno0-deno1)**2))
    print("grads: ",th.mean((grads0-grads1)**2))


def main():

    # -- config --
    cfg = edict()
    cfg.seed = 123
    cfg.device = "cuda:0"
    cfg.dname = "set8"
    cfg.dset = "val"
    cfg.isize = "128_128"
    cfg.vid_names = ["sunflower","tractor","snowboard"]
    cfg.ntype = "g"
    cfg.sigma = 0.1
    cfg.nframes = 3
    cfg.stride0 = 1
    cfg.ws = 9
    cfg.wt = 1
    cfg.wr = 1
    cfg.ps = 1
    cfg.k = 5
    cfg.k_agg = 10
    cfg.dist_type = "l2"
    cfg.qk_frac = 0.5
    cfg.attn_proj_version = "v1"
    cfg.topk_mode = "each"
    cfg.embed_dim = 16
    cfg.arch_depth = [1,2,4]
    cfg.arch_nheads = [1,4,12]
    cfg.block_version = ["v9","v9","v11"]
    cfg.itype = "int"
    cfg.use_spynet = True
    cfg.dd_in = 3
    cfg.nres_per_block = 2
    cfg.search_menu_name = "first"
    cfg.agg_name = "stack_conv"
    run(cfg)

if __name__ == "__main__":
    main()
