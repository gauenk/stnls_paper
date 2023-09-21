
import torch as th
import numpy as np
import cv2

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import center_crop
from torchvision.utils import save_image,make_grid

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
    return get_data_set(dcfg)

def get_data_set(dcfg):
    data,loaders = data_hub.sets.load(dcfg)
    indices = data_hub.filter_subseq(data[dcfg.dset],dcfg.vid_name,
                                     dcfg.frame_start,dcfg.frame_end)
    device = "cuda:0"
    nvid = data[dcfg.dset][indices[0]]['noisy'].to(device)/255.
    vid = data[dcfg.dset][indices[0]]['clean'].to(device)/255.
    return vid,nvid

def run_exps(cfg,dcfg):

    # -- get video --
    set_seed(dcfg.seed)
    vid,nvid = get_data(dcfg)

    # -- get sims --
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
    stacking = stnls.tile.NonLocalStack(1,cfg.stride0,
                                        itype_fwd="float",itype_bwd="float")
    flows = flow.orun(nvid,cfg.flow,ftype="cv2")
    acc_flows = stnls.nn.accumulate_flow(flows.fflow,flows.bflow)
    dists,inds = search_p.paired_vids(nvid,nvid,acc_flows,cfg.wt,skip_self=True)
    ones = th.ones_like(dists)
    # M = inds.shape[2]//2
    # print(inds[0,0,M,...,0])
    stack = stacking(vid,ones,inds)[:,0]

    # -- compute sims --
    loss = th.mean((vid[:,None] - stack)**2,dim=(0,3,4,5))
    # print(cfg.name,cfg.flow,cfg.ws)
    # print(loss.T)

    # -- save example --
    flow_s = "withflow" if cfg.flow else "noflow"
    ws_s = "yS" if cfg.ws > 1 else "nS"
    args = (dcfg.vid_name,flow_s,ws_s)
    root = Path("output/figures/compare_sims/%s/%s_%s/" % args)
    if not(root.exists()): root.mkdir(parents=True)
    psnrs = compute_psnrs(stack[:,[cfg.ai],cfg.ti],vid[:,None,cfg.ti])
    pi = cfg.ti+1 if cfg.ai == 0 else cfg.ti-1


    # -- cropping --
    sH,eH = cfg.sH,cfg.eH
    sW,eW = cfg.sW,cfg.eW
    # sH = 256-128
    # eH = sH + 256
    # sW = 256-128
    # eW = sW + 256
    # print(sH,eH,sW,eW)
    vid = vid[...,sH:eH,sW:eW]
    stack = stack[...,sH:eH,sW:eW]
    # print(vid.shape)

    # -- compute psnr --
    psnrs = compute_psnrs(stack[:,[cfg.ai],cfg.ti],vid[:,None,cfg.ti])

    # -- sizing --
    tH,tW = 384,384
    ref = ensure_size(vid[0,cfg.ti],tH,tW)
    prop = ensure_size(vid[0,pi],tH,tW)
    aligned = ensure_size(stack[0,cfg.ai,cfg.ti],tH,tW)

    # # vid_io.save_video(stack[:,:,ti],root,"align",itype="png")
    # delta = th.abs(stack[:,:,ti] - vid[:,None,ti])
    # delta = th.abs(stack[:,:,ti] - vid[:,None,ti]).mean(-3,keepdim=True)
    # print("dmax: ",delta.max().item())
    # delta /= 0.8273#delta.max()
    # if delta.max() > 1:
    #     delta -= delta.min()
    #     delta = delta / delta.max()
    # psnrs = compute_psnrs(stack[:,:,ti],vid[:,None,ti].repeat(1,2*cfg.k,1,1,1))
    # print("psnrs: ",psnrs)
    # print("dmax: ",delta.max().item())
    # # vid_io.save_video(delta,root,"delta",itype="png")


    # # -- save center crops --
    # H,W = vid.shape[-2:]
    # cH,cW = 128,128
    # sH = 128+32
    # eH = sH + 128+64-32
    # sW = 128#+128
    # eW = sW + 128+64-32

    # # nlstack = stack[:,:,ti,:,sH:eH,sW:eW]
    # # vid_io.save_video(nlstack,root,"cc",itype="png")

    # # -- create gt reference --
    root = Path("output/figures/compare_sims/%s/gt/" % dcfg.vid_name)
    # if not(root.exists()): root.mkdir(parents=True)
    # # vid_cc = vid[:,:,:,sH:eH,sW:eW]
    vid_io.save_video(vid,root,"cc",itype="png")

    if "tractor" in dcfg.vid_name:
        val = 20
        ref = increase_brightness(ref, value=val)
        prop = increase_brightness(prop, value=val)
        aligned = increase_brightness(aligned, value=val)

    return ref,prop,aligned,psnrs[0].item()

def increase_brightness_vid(vid, value=30):
    T = vid.shape[0]
    for t in range(T):
        vid[t] = increase_brightness(vid[t], value=30)
    return vid

def increase_brightness(img, value=30):

    device = img.device
    img = img.cpu().numpy()
    img = rearrange(img,'c h w -> h w c')
    img /= img.max()
    img *= 255
    img = np.clip(img,0,255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

    img = th.from_numpy(img).to(device)
    img = rearrange(img,'h w c -> c h w')*1.
    img /= img.max()

    return img

def run_grid(dcfg,vid_names):
    ps = 3
    ws = 11
    s1 = 0.5
    grid = []
    psnrs = []
    for vid_name in vid_names:
        vinfo = vid_name.split(":")
        dcfg.vid_name = vinfo[0]
        align_index = int(vinfo[1])

        if len(vinfo) > 2:
            sH,eH,sW,eW = [int(s) for s in vinfo[2].split("-")]
        else:
            sH,eH,sW,eW = 0,512,0,512
        cfgs = [edict({"name":"stnls","ps":ps,"ws":ws,"full_ws":False,
                       "wt":1,"k":1,"stride0":1,"stride1":s1,"flow":False,
                       "ti":1,"ai":align_index,"sH":sH,"eH":eH,"sW":sW,"eW":eW}),
                edict({"name":"stnls","ps":1,"ws":1,"full_ws":False,
                       "wt":1,"k":1,"stride0":1,"stride1":.1,"flow":True,
                       "ti":1,"ai":align_index,"sH":sH,"eH":eH,"sW":sW,"eW":eW}),
                edict({"name":"stnls","ps":ps,"ws":ws,"full_ws":False,
                       "wt":1,"k":1,"stride0":1,"stride1":s1,"flow":True,
                       "ti":1,"ai":align_index,"sH":sH,"eH":eH,"sW":sW,"eW":eW})]

        sims,psnrs_v = [],[]
        for cfg in cfgs:
            ref,prop,aligned,psnr = run_exps(cfg,dcfg)
            sims.append(aligned)
            psnrs_v.append(psnr)
        vstack = th.stack(sims+[ref,prop,])
        grid.append(vstack)
        psnrs.append(th.tensor(psnrs_v))
    grid = th.stack(grid)
    psnrs = th.stack(psnrs)
    print(grid.shape)
    print(psnrs)
    return grid,psnrs

def ensure_size(img,tH,tW):
    img = TF.resize(img,(tH,tW),InterpolationMode.BILINEAR)
    return img

def main():

    # fn = "/home/gauenk/Documents/data/davis/DAVIS/ImageSets/2017/train-val.txt"
    # vid_names = np.loadtxt(fn,str)
    vid_names = ["cat-girl:0","classic-car:0","color-run:0",
                 "dog-gooses:0","drone:0","hockey:0",
                 "horsejump-low:0","kid-football:0","lady-running:0",
                 "lindy-hop:0","lucia:0","motocross-bumps:0",
                 "motorbike:0","paragliding:0","scooter-board:0",
                 "scooter-grey:0","skate-park:0","snowboard:0",
                 "stroller:0","stunt:0","surf:0",
                 "swing:0","tennis:0","tractor-sand:0",
                 "tuk-tuk:0","upside-down:0","walking:0"]
    best_names = ["kid-football:0:18-176-8-170",
                  "color-run:0:192-256-300-364",
                  "walking:1:192-320-128-256",
                  "scooter-board:1:112-256-220-362",
                  "scooter-gray:1:64-192-160-288",
                  "swing:0:96-292-192-388",
                  "stroller:0:128-384-64-320",
                  "dog-gooses:0:80-176-128-224",
                  "drone:0:128-320-128-320",
                  "hockey:0:64-192-224-352",
                  "lady-running:0:64-256-64-256",
                  "lindy-hop:0:64-256-320-512",
                  "tennis:0:96-292-96-292",
                  "tuk-tuk:1:64-172-220-328",
                  ]
    vid_names = ["cat-girl:0:256-512-256-512",
                 "classic-car:0:256-384-256-384",
                 "color-run:0:192-256-300-364",
                 "dog-gooses:0:80-176-128-224",
                 "drone:0:128-320-128-320",
                 "hockey:0:64-192-224-352",
                 "horsejump-low:0:100-420-192-512",
                 "kid-football:0:18-176-8-170"
                 "lady-running:0:0-256-0-256",
                 "lindy-hop:0:64-256-320-512",
                 "lucia:0:96-256-96-256",
                 "motocross-bumps:0:128-384-0-256",
                 "motorbike:0:64-320-0-256",
                 "paragliding:0:128-384-256-512",
                 "scooter-board:1:112-256-220-362",
                 "scooter-gray:1:64-138-180-254",
                 "skate-park:1:240-368-288-416","snowboard:0:32-160-128-256",
                 "stroller:0:128-384-64-320","stunt:1:128-384-64-320",
                 "surf:0:64-320-0-256","swing:0:106-198-218-314",
                 "tennis:0:96-292-96-292","tractor-sand:0:96-292-96-292",
                 "tuk-tuk:1:64-172-220-328",
                 "upside-down:1:128-384-64-320",
                 "walking:1:192-320-128-256"]
    # vid_names = ["dog-gooses:0:80-176-128-224",
    #              "hockey:0:64-192-224-352",]
    # vid_names = best_names
    vid_names = ["color-run:0:192-256-300-364",
                 "kid-football:0:18-176-8-170",
                 "tennis:0:96-292-96-292",
                 "scooter-gray:1:44-158-160-274",
                 # "swing:0:106-198-218-314",
                 # "tennis:0:96-292-96-292",
                 "lindy-hop:0:74-174-340-440",
                 "walking:1:192-320-128-256",
                 # "scooter-board:1:112-256-220-362",
                 "stroller:0:138-266-74-202",
                 # "dog-gooses:0:80-176-128-224",
                 "tractor-sand:0:128-292-128-292",
    ]
    fstart = 0
    fend = fstart + 5 - 1
    dcfg = edict({"dname":"davis","dset":"tr","vid_name":"","sigma":15,
                  "nframes":5,"frame_start":fstart,"frame_end":fend,
                  "isize":"512_512","seed":123})
    grid,psnrs = run_grid(dcfg,vid_names)
    print(grid.shape)
    nrow = grid.shape[0]
    grid = grid.transpose(0,1).flatten(0,1)
    print(grid.shape)
    grid = make_grid(grid,nrow=nrow)
    save_image(grid,'grid_a.png')

if __name__ == "__main__":
    main()

