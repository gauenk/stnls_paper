"""

   Is there a correlation when the GDA target is the grid search?


"""


import collections
import torch as th
import numpy as np
import pandas as pd
from torchvision.transforms.functional import center_crop

import frame2frame
from frame2frame.nb2nb_loss import generate_mask_pair,generate_subimages
import stnls
import data_hub
from dev_basics import flow
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange
from dev_basics.utils.misc import set_seed
from dev_basics.utils import vid_io
from dev_basics.utils.metrics import compute_psnrs
import cache_io


from natten.functional import natten2dav, natten2dqkrpb

# -- plotting --
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse

# -- pairing --
# from stnls.search.paired_utils import paired_vids
from stnls.search.utils import paired_vids

# -- bench --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt

from stnls_paper.trte_align.align_model import AlignModel

def get_data(dcfg):
    dcfg.rand_order = False
    data,loaders = data_hub.sets.load(dcfg)
    indices = data_hub.filter_subseq(data[dcfg.dset],dcfg.vid_name,
                                     dcfg.frame_start,dcfg.frame_end)
    device = "cuda:0"
    # print(indices[0])
    # print(data[dcfg.dset].groups[indices[0]])
    vid = data[dcfg.dset][indices[0]]['clean'].to(device)/255.
    nvid = data[dcfg.dset][indices[0]]['noisy'].to(device)/255.
    # print(th.mean((nvid-vid)**2).sqrt()*255.)
    return vid[None,:],nvid[None,:]

def run_gda(nvid,acc_flows,epoch):
    cfg = edict()
    cfg.stride0 = 1
    cfg.attn_size = 1
    spynet_path = ""
    if epoch < 0: epoch = -2
    chkpt_path = "output/deno/train/checkpoints/77582d15-8430-4ca1-86ca-25a597249c2c/77582d15-8430-4ca1-86ca-25a597249c2c-epoch=%02d.ckpt" % epoch
    model = AlignModel(cfg,"gda",spynet_path)
    chkpt = th.load(chkpt_path)
    new_state_dict = collections.OrderedDict()
    for k, v in chkpt['model_state_dict'].items():
        name = k.replace("module.", '') # remove `module.`
        new_state_dict[name] = v
    # print(chkpt['model_state_dict'])
    # print(new_state_dict)
    model.load_state_dict(new_state_dict)
    nvid= nvid.cuda()
    model = model.cuda()
    # -- run only gda --
    with th.no_grad():
        flows_k = model.model.paired_vids(nvid, nvid, acc_flows,
                                          model.wt, skip_self=True)[1]
    return flows_k

def run_stnls(nvid,acc_flows,ws,wt,ps,s0,s1,full_ws=False):
    k = 1
    search_p = stnls.search.PairedSearch(ws,ps,k,
                                         nheads=1,dist_type="l2",
                                         stride0=s0,stride1=s1,
                                         self_action=None,use_adj=False,
                                         full_ws=full_ws,itype="float")
    _,flows = search_p.paired_vids(nvid,nvid,acc_flows,wt,skip_self=True)
    return flows

def compute_epe(a,b):
    return th.mean((a-b).abs())

def compute_corr(dir0,dir1):
    m0 = dir0.mean()
    m1 = dir1.mean()
    num = th.sum( (dir0-m0)*(dir1-m1) )
    den0 = th.sum( (dir0-m0)**2)
    den1 = th.sum( (dir1-m1)**2)
    den = (den0*den1).sqrt()
    corr = num / den
    return corr

def main():

    # -- config --
    ws = 11
    wt = 1
    ps = 3
    s0 = 1
    s1 = 1.
    dname = "davis"
    dset = "tr"
    vid_name = "bear"
    dcfg = edict({"dname":dname,"dset":dset,"vid_name":vid_name,"sigma":15.,
                  "nframes":5,"frame_start":0,"frame_end":4,
                  "isize":None,"seed":123})
    th.manual_seed(dcfg.seed)

    # -- get video --
    vid,nvid = get_data(dcfg)
    flows = flow.orun(nvid,True,ftype="cv2")
    acc_flows = stnls.nn.search_flow(flows.fflow,flows.bflow,wt,s0)

    # -- get gt flows --
    flows_gt = run_stnls(nvid,acc_flows,ws,wt,ps,s0,s1)

    # -- compute correlations --
    epe = []
    corr = []
    nepochs = 3
    for epoch in range(0,nepochs):
        flows_t0 = run_gda(nvid,acc_flows,epoch-1)
        flows_t1 = run_gda(nvid,acc_flows,epoch)
        dir_tr = flows_t1 - flows_t0
        dir_gt = flows_gt - flows_t0
        corr_e = compute_corr(dir_tr,dir_gt)
        epe_e = compute_epe(flows_t1,flows_gt)
        corr.append(corr_e)
        epe.append(epe_e)
    print(corr)
    print(epe)

if __name__ == "__main__":
    main()
