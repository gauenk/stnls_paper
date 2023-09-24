
import torch as th
import numpy as np
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

def get_data(dcfg):
    # return get_data_example(dcfg)
    return get_data_set(dcfg)

def get_data_example(dcfg):
    root = Path("output/figures/crop_cat_chicken")
    device = "cuda:0"
    vid = vid_io.read_video(root).to(device)/255.
    nvid = vid + dcfg.sigma/255. * th.randn_like(vid)
    # print(vid.shape,nvid.shape)
    return vid,nvid

def get_data_set(dcfg):
    data,loaders = data_hub.sets.load(dcfg)
    indices = data_hub.filter_subseq(data[dcfg.dset],dcfg.vid_name,
                                     dcfg.frame_start,dcfg.frame_end)
    print(indices)
    # vid = data[dcfg.dset][indices[0]]['clean']
    device = "cuda:0"
    nvid = data[dcfg.dset][indices[0]]['noisy'].to(device)/255.
    vid = data[dcfg.dset][indices[0]]['clean'].to(device)/255.
    print(th.mean((nvid-vid)**2).item())
    return vid,nvid

def run_exps(cfg,dcfg):


    # -- get video --
    set_seed(dcfg.seed)
    vid,nvid = get_data(dcfg)
    # print(vid.numel())
    # vid = th.arange(vid.numel()).reshape(vid.shape).to(device)*1.
    # print(vid.shape)

    # -- get sims --
    if cfg.name == "nb2nb":
        mask0,mask1 = generate_mask_pair(vid)
        vid0 = generate_subimages(vid, mask0)
        vid1 = generate_subimages(vid, mask1)
        vid = vid[...,::2,::2].contiguous()
    elif cfg.name == "n2n":
        mask0,mask1 = generate_mask_pair(vid)
        vid0 = generate_subimages(vid, mask0)
        vid1 = generate_subimages(vid, mask1)
        vid = vid[...,::2,::2].contiguous()
    elif cfg.name == "stnls":
        vid = vid[None,:].contiguous()
        nvid = nvid[None,:].contiguous()
        search = stnls.search.NonLocalSearch(cfg.ws,cfg.wt,cfg.ps,cfg.k,
                                             nheads=1,dist_type="l2",
                                             stride0=cfg.stride0,
                                             anchor_self=True,use_adj=False,
                                             full_ws=cfg.full_ws)
        search_p = stnls.search.PairedSearch(cfg.ws,cfg.ps,cfg.k,
                                             nheads=1,dist_type="l2",
                                             stride0=cfg.stride0,
                                             stride1=cfg.stride1,
                                             anchor_self=False,use_adj=False,
                                             full_ws=cfg.full_ws,
                                             full_ws_time=cfg.full_ws,
                                             itype_fwd="float",itype_bwd="float")
        stacking = stnls.tile.NonLocalStack(cfg.ps_stack,cfg.stride0,
                                            itype_fwd="float",itype_bwd="float")
        flows = flow.orun(nvid,cfg.flow,ftype="cv2")
        print(th.mean(flows.fflow**2).item(),th.mean(flows.bflow**2).item())
        # dists,inds = search(nvid,nvid,flows.fflow,flows.bflow)
        acc_flows = stnls.nn.accumulate_flow(flows.fflow,flows.bflow)
        dists,inds = search_p.paired_vids(nvid,nvid,acc_flows,cfg.wt,skip_self=True)
        # print(dists.shape,inds.shape)
        # print(inds[0,0,60])
        ones = th.ones_like(dists)
        stack = stacking(vid,ones,inds)[:,0]
        # vid0 = stack[:,0,1]
        # vid1 = stack[:,0,2]

        # print("inds.shape: ",inds.shape)
        # print(inds[0,0,0,0])
        # print(inds[0,0,1,0])
        # use_adj = False#cfg.ps//2
        # # adj = 0
        # unfold = stnls.UnfoldK(cfg.ps,use_adj=use_adj,reflect_bounds=True)
        # adj = 0#cfg.ps//2
        # fold = stnls.iFoldz(vid.shape,use_adj=use_adj)
        # inds = inds[:,0]
        # # print("inds.shape: ",inds.shape)
        # patches0 = unfold(vid,inds[...,[4],:])
        # patches1 = unfold(vid,inds[...,[5],:])
        # T = vid.shape[0]
        # print("-"*10 + " patches " +"-"*10)
        # print(patches0[0,0,0,0])
        # print("-"*10 + " video " +"-"*10)
        # print(vid[0,0,:,:5,:5])
        # print("patches0.shape: ",patches0.shape)
        # print("vid.shape: ",vid.shape)
        # p2 = cfg.ps//2
        # patches0 = patches0[...,p2,p2][...,None,None]
        # patches1 = patches1[...,p2,p2][...,None,None]

        # print("patches0.shape: ",patches0.shape)
        # H,W = vid.shape[-2:]
        # patches0[1:] = 0
        # vid0 = rearrange(patches0,'(t h w) k 1 c 1 1 -> k t c h w',h=H,w=W)[0]
        # vid1 = rearrange(patches1,'(t h w) k 1 c 1 1 -> k t c h w',h=H,w=W)[0]

        # patches0[2:] = 0
        # patches1[2:] = 0
        # vid0,wvid0 = fold(patches0)
        # vid1,wvid1 = fold(patches1)

        # # -- normalize --
        # vid0 = vid0 / wvid0
        # vid1 = vid1 / wvid1

        # -- view --
        # print("-"*10 + " video0 " +"-"*10)
        # print(vid0[0,:,:5,:5])


    # -- compute sims --
    loss = th.mean((vid[:,None] - stack)**2,dim=(0,3,4,5))
    # loss0 = th.mean((vid - vid0)**2).item()
    # loss1 = th.mean((vid - vid1)**2).item()
    print(cfg.name,cfg.flow,cfg.ws)
    print(loss.T)

    # -- save example --
    flow_s = "withflow" if cfg.flow else "noflow"
    ws_s = "yS" if cfg.ws > 1 else "nS"
    args = (dcfg.vid_name,flow_s,ws_s)
    root = Path("output/figures/compare_sims/%s/%s_%s/" % args)
    if not(root.exists()): root.mkdir(parents=True)
    ti = 1
    vid_io.save_video(stack[:,:,ti],root,"align",itype="png")
    delta = th.abs(stack[:,:,ti] - vid[:,None,ti])
    delta = th.abs(stack[:,:,ti] - vid[:,None,ti]).mean(-3,keepdim=True)
    print("dmax: ",delta.max().item())
    delta /= 0.8273#delta.max()
    if delta.max() > 1:
        delta -= delta.min()
        delta = delta / delta.max()
    psnrs = compute_psnrs(stack[:,:,ti],vid[:,None,ti].repeat(1,2*cfg.k,1,1,1))
    print("psnrs: ",psnrs)
    print("dmax: ",delta.max().item())
    vid_io.save_video(delta,root,"delta",itype="png")


    # -- save center crops --
    H,W = vid.shape[-2:]
    cH,cW = 128,128
    sH = 128+32
    eH = sH + 128+64-32
    sW = 128#+128
    eW = sW + 128+64-32

    nlstack = stack[:,:,ti,:,sH:eH,sW:eW]
    vid_io.save_video(nlstack,root,"cc",itype="png")

    # -- create gt reference --
    root = Path("output/figures/compare_sims/%s/gt/" % dcfg.vid_name)
    if not(root.exists()): root.mkdir(parents=True)
    vid_cc = vid[:,:,:,sH:eH,sW:eW]
    vid_io.save_video(vid_cc,root,"cc",itype="png")

def main():

    fn = "/home/gauenk/Documents/data/davis/DAVIS/ImageSets/2017/train-val.txt"
    vid_names = np.loadtxt(fn,str)
    # tough ones; dance-jump, dancing, dog-agility
    vid_names = ["cat-girl","classic-car","color-run","dog-gooses","drone","hockey","horsejump-low","kid-football","lady-running","lindy-hop","lucia","motorcross-bumps","motorbike","paragliding","scooter-board","scooter-grey","skate-park","snowboard","stroller","stunt","surf","swing","tennis","tractor-sand","tuk-tuk","upside-down","walking"]
    vid_names = [vid_names[0]]
    for vid_name in vid_names:
        fstart = 0
        fend = fstart + 5 - 1
        dcfg = edict({"dname":"davis","dset":"tr","vid_name":vid_name,"sigma":0.,
                      "nframes":5,"frame_start":fstart,"frame_end":fend,
                      "isize":"512_512","seed":123})
        ps = 3
        ps_stack = 3
        ws = 11
        s0 = 2
        s1 = 0.5
        cfgs = [edict({"name":"stnls","ps":ps,"ps_stack":ps_stack,
                       "ws":ws,"full_ws":False,
                       "wt":1,"k":1,"stride0":1,"stride1":s1,"flow":False}),
                edict({"name":"stnls","ps":1,"ps_stack":1,
                       "ws":1,"full_ws":False,
                       "wt":1,"k":3,"stride0":1,"stride1":.1,"flow":True}),
                edict({"name":"stnls","ps":ps,"ps_stack":ps_stack,
                       "ws":ws,"full_ws":False,
                       "wt":1,"k":1,"stride0":s0,"stride1":s1,"flow":True})]
        for cfg in cfgs:
            results = run_exps(cfg,dcfg)

if __name__ == "__main__":
    main()
