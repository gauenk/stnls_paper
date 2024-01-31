
# -- python imports --
import os
import numpy as np
import torch as th
import pandas as pd
from dev_basics.utils.misc import set_seed,optional
from easydict import EasyDict as edict
from dev_basics.utils import vid_io

# -- networks --
import importlib

# -- data --
import data_hub

# -- chunking --
from dev_basics import net_chunks

# -- optical flow --
from dev_basics import flow

# -- recording --
import cache_io

# -- losses --
from frame2frame import get_loss_fxn

# -- metrics --
from dev_basics.utils.metrics import compute_psnrs,compute_ssims,compute_strred
from dev_basics.utils.misc import optional

import stnls
class SLIC(th.nn.Module):
    def __init__(self,lr=1e-3,niters=6000,k=5,ps=3,ws=9,wt=1,stride0=2,
                 use_flow=True,flow_method="cv2"):
        super().__init__()
        self.k = k
        self.ps = ps
        self.ws = ws
        self.wt = wt
        self.stride0 = stride0
        self.use_flow = use_flow
        self.flow_method = flow_method
        self.nmz_bwd = False
        self.niters = niters
        self.lr = lr

    def get_flows(self,vid):
        flows = flow.orun(vid,self.use_flow,ftype=self.flow_method)
        wt,stride0 = self.wt,self.stride0
        flows = stnls.nn.search_flow(flows.fflow,flows.bflow,wt,stride0)
        return flows

    def get_search_fxns(self):
        ws,wt,k,ps = self.ws,self.wt,self.k,self.ps
        search = stnls.search.NonLocalSearch(ws,wt,ps,-1,
                                             nheads=1,dist_type="l2",
                                             stride0=self.stride0,
                                             self_action="anchor",
                                             full_ws=True,
                                             topk_mode="each",
                                             normalize_bwd=self.nmz_bwd,
                                             itype="float")
        k,wr,kr = -1,1,1.
        refine = stnls.search.RefineSearch(ws,wt,wr,k,kr,ps,nheads=1,
                                           dist_type="l2",stride0=self.stride0,
                                           normalize_bwd=self.nmz_bwd,
                                           itype="float")
        # stack = stnls.agg.NonLocalStack(ps,stride0,itype="float")
        return search,refine#,stack

    def forward(self,vid,anno_gt):

        # -- init --
        search,refine = self.get_search_fxns()
        flows = self.get_flows(vid)
        flows = flows.round()

        # -- init slic --
        centroids = vid.clone()
        stride0 = search.stride0
        W_t,ws = 2*search.wt+1,search.ws
        zflows = th.zeros_like(flows[:,:,0])
        flows = th.stack([zflows,flows],2)
        assert flows.shape[2] == W_t
        res = 1000.
        term_tol = 1e-3
        niters = 10
        iter_i = 0

        while (res > term_tol) or (iter_i >= niters):

            # -- compute distances  --
            dists,flows_k = search(centroids,vid,flows)
            B,HD,T,nH,nW = dists.shape[:5]

            # -- scatter searched values to source --
            labels = stnls.scatter.get_indices(T,H,W,stride0,wt,ws)
            dists = dists.view(B,HD,T,1,nH,nW,-1)
            dists = stnls.scatter.run_scatter(dists,flows,labels)
            # dists.shape = (B,HD,T,1,H,W,S)

            flows_k = rearrange(flows_k,'b hd t h w k tr -> b hd t tr h w k')
            flows_k = stnls.scatter.run_scatter(flows_k,flows,labels)
            # flows.shape = (B,HD,T,3,H,W,S)

            # -- select top-k=1 at each source --
            dists,flows,labels = select_topk(dists,flows_k,labels,k=1)

            # -- masked weighted sum --
            centroids_prev = centroids
            centroids = stnls.nn.labeled_mean(vid,labels)
            flows = stnls.nn.labeled_mean(flows,labels)

            # -- terminate condition --
            res = th.mean((centroids - centroids_prev)**2).item()
            iter_i += 1

# def unfold_windows(dists,flows,

def img_grads(vid):
    from th.nn.functional import conv2d
    B = vid.shape[0]
    vid = rearrange(vid,'b t c h w -> (b t) c h w')
    weight = th.FloatTensor([[1, 2, 1],
                             [0,0,0],
                             [-1,-2,-1]]).to(vid.device)
    Gx = conv2d(vid,weight,padding=1)
    Gy = conv2d(vid,weight,padding=1)
    print("Gx.shape: ",Gx.shape)
    print("Gy.shape: ",Gx.shape)
    G = th.stack([Gx,Gy],-3)
    G = rearrange(G,'(b t) c h w -> b t c h w',b=B)
    return G

def get_videos(cfg):

    # -- load from files --
    device = "cuda:0"
    data,loaders = data_hub.sets.load(cfg)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,0,-1)
    assert len(indices) == 1,"Must only be one video subsequence."
    _vid = data[cfg.dset][indices[0]]['clean'][None,:].to(device)/255.
    _anno = data[cfg.dset][indices[0]]['annos'][None,:].to(device).round()

    # -- save --
    # fft_n = th.fft.fftshift(th.fft.rfft2(_noisy[:,:10].mean(-3,keepdim=True)))
    # fft_c = th.fft.fftshift(th.fft.rfft2(_clean[:,:10].mean(-3,keepdim=True)))
    # # fft_n_abs = fft_n.abs()
    # # fft_n_abs = fft_n.angle()
    # # print(fft_n.shape)
    # vid_io.save_video(20*th.log10(fft_n[:,:10].abs()),"output/saved_examples","noisy_abs")
    # vid_io.save_video(fft_n[:,:10].angle(),"output/saved_examples","noisy_angle")
    # vid_io.save_video(20*th.log10(fft_c[:,:10].abs()),"output/saved_examples","clean_abs")
    # vid_io.save_video(fft_c[:,:10].angle(),"output/saved_examples","clean_angle")
    # vid_io.save_video(_noisy[:,:10],"output/saved_examples","noisy")
    # vid_io.save_video(_clean[:,:10],"output/saved_examples","clean")
    # exit()

    # # -- add noise channel --
    # if optional(cfg,"dd_in",3) == 4:
    #     _noisy = append_sigma(_noisy,cfg.sigma)

    # -- dev subsampling --
    # print("_anno.shape: ",_anno.shape)
    _vid = _vid[:,:6]
    _anno = _anno[:,:6]

    # -- info --
    print(_anno.shape)

    # -- split --
    vid,anno = split_vids(_vid,_anno,cfg.num_tr_frames)

    return vid,anno

def append_sigma(noisy,sigma):
    if noisy.shape[-3] == 4: return noisy
    sigma_map = th.ones_like(noisy[:,:,:1])*(sigma/255.)
    noisy = th.cat([noisy,sigma_map],2)
    return noisy

def split_vids(_vid,_anno,num_tr):
    vid,anno = edict(),edict()
    vid.tr = _vid[:,:num_tr].contiguous()
    vid.te = _vid[:,num_tr:].contiguous()
    anno.tr = _anno[:,:num_tr].contiguous()
    anno.te = _anno[:,num_tr:].contiguous()
    return vid,anno

def load_model(cfg):
    device = "cuda:0"
    net = SLIC().to(device).train()
    return net

def get_scheduler(cfg,name,optim):
    lr_sched = th.optim.lr_scheduler
    if name in [None,"none"]:
        return lr_sched.LambdaLR(optim,lambda x: x)
    elif name in ["cosa"]:
        nsteps = cfg.seq_nepochs*cfg.num_tr_frames
        scheduler = lr_sched.CosineAnnealingLR(optim,T_max=nsteps)
        return scheduler
    else:
        raise ValueError(f"Uknown scheduler [{name}]")

def run_training(cfg,model,noisy,clean):


    # -- get loss --
    if cfg.loss_type != "none":
        # -- optimizer --
        optim = th.optim.Adam(model.parameters(),lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
        scheduler = get_scheduler(cfg,optional(cfg,"scheduler_name",None),optim)
        assert noisy.shape[1] == cfg.num_tr_frames,"Must be equal."

        # -- init --
        loss_fxn = get_loss_fxn(cfg,cfg.loss_type)

        # -- run --
        train_info = loss_fxn(model,optim,scheduler,noisy,clean)
    else:
        # -- skip --
        train_info = {}

    # -- test on training data --
    test_info = run_testing(cfg,model,noisy,clean)

    # -- info --
    info = edict()
    for key in train_info:
        info[key] = train_info[key]
    for key in test_info:
        info["tr_%s"%key] = test_info[key]

    return info

def run_testing(cfg,model,vid,anno):

    # -- denoised output --
    model = model#.eval()
    chunk_cfg = net_chunks.extract_chunks_config(cfg)
    fwd_fxn0 = lambda vid,flows=None: model(vid,anno)
    # fwd_fxn0 = lambda vid,flows=None: run_updates(model,vid)
    fwd_fxn = net_chunks.chunk(chunk_cfg,fwd_fxn0)
    # with th.no_grad():

    anno_est = fwd_fxn(vid) # enable grad
    anno_est = anno_est.round()

    mse = th.mean((anno - anno_est)**2).item()
    # print("deno.shape: ",deno.shape,vid.shape)
    psnrs_0 = compute_psnrs(anno,anno_est,div=1.)
    psnrs_1 = compute_psnrs(1-anno,anno_est,div=1.)
    # psnrs_noisy = compute_psnrs(noisy[...,:3,:,:],vid,div=1.)
    # ssims = compute_ssims(deno,vid,div=1.)
    # ssims_noisy = compute_ssims(noisy[...,:3,:,:],vid,div=1.)
    # strred = compute_strred(deno,vid,div=1.)
    # # print(psnrs,psnrs_noisy)

    # -- save --
    vid_io.save_video(anno_est,"output/saved_annos","anno_est")
    vid_io.save_video(anno,"output/saved_annos","anno_gt")

    # -- info --
    info_te = edict()
    info_te.mse = mse
    info_te.psnrs_0 = psnrs_0.mean().item()
    info_te.psnrs_1 = psnrs_1.mean().item()
    return info_te

def run(cfg):

    # -- init --
    set_seed(cfg.seed)
    # set_pretrained_path(cfg)

    # -- read data --
    vid,anno = get_videos(cfg)

    # -- read model --
    model = load_model(cfg)

    # -- run testing --
    # info_tr = run_training(cfg,model,vid.tr,anno.tr)
    info_tr = {}
    info_te = run_testing(cfg,model,vid.te,anno.te)

    # -- create results --
    results = edict()
    for k,v in info_tr.items():
        results[k] = v
    for k,v in info_te.items():
        assert not(k in results)
        results[k] = v

    print(results)
    print(pd.DataFrame(results))
    exit()
    return results

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#      Separate Config Grids
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def grid_v0():
    exps = {"group0":{"cut_name":["v0"]}}
    return exps

def collect_grids(base):
    # grids = [f2f_grid,f2f_plus_grid,stnls_grid,none_grid]
    # grids = [none_grid,stnls_grid,f2f_grid]
    # grids = [none_grid,stnls_grid]
    grids = [grid_v0]
    # grids = [none_grid,]#stnls_grid]
    cfgs = []
    for grid in grids:
        exps = base | grid()
        # if grid != none_grid: exps = exps | learn
        cfgs += cache_io.exps.load_edata(exps)
    return cfgs


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#         Launching
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def main():

    # -- init --
    print("PID: ",os.getpid())

    # -- config --
    base = {
        # v0.1
        "group10":{"tag":["v0.000"],"seed":[123]},
        "group11":{"vid_name":[[
            "bear",
        ]],"dname":["davis_anno"],"dset":["tr"],"sigma":[0.1]},
        "group12":{"num_tr_frames":[3],"iscale":[0.25]},
        # "group12":{"net_module":["frame2frame"],
        #            "net_name":["dncnn"],
        #            "dd_in":[3],
        #            # "net_name":["fdvd"],
        #            # "dd_in":[4],
        #            },
    }
    # base_learn = {
    #     "group14": {"lr":[1.001e-3],"weight_decay":[1e-8],
    #                 "seq_nepochs":[[70]],"scheduler_name":["cosa"],
    #                 "spatial_chunk_size":[256],"spatial_chunk_overlap":[0.1],
    #                 "temporal_chunk_size":[5],"unsup_isize":["156_156"],
    #                 "nbatch_sample":[10]}
    # }
    # # base['listed100'] = sigma_grids()
    # base['listed100'] = sr_grids()
    exps = collect_grids(base)

    # -- run --
    results = cache_io.run_exps(exps,run,proj_name="graph_cut",
                                name=".cache_io/graph_cut",
                                records_fn=".cache_records/graph_cut.pkl",
                                records_reload=True,
                                enable_dispatch="slurm",use_wandb=True)
    if len(results) == 0:
        print("No results")
        return

    # -- view --
    print(results)


if __name__ == "__main__":
    main()
