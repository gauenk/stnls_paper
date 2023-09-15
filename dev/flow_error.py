
import torch as th
import numpy as np
from torchvision.transforms.functional import center_crop
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.utils import draw_bounding_boxes
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
    # return get_data_example(dcfg)
    vid,nvid = get_data_set(dcfg)
    # vid = vid[None,:].contiguous()
    # nvid = nvid[None,:].contiguous()
    return vid,nvid
def get_data_set(dcfg):
    data,loaders = data_hub.sets.load(dcfg)
    indices = data_hub.filter_subseq(data[dcfg.dset],dcfg.vid_name,
                                     dcfg.frame_start,dcfg.frame_end)
    print(indices)
    device = "cuda:0"
    nvid = [data[dcfg.dset][i]['noisy'].to(device)/255. for i in indices]
    vid = [data[dcfg.dset][i]['clean'].to(device)/255. for i in indices]

    sH,sW,sSize = dcfg.sH,dcfg.sW,dcfg.sSize
    eH,eW = sH+sSize,sW+sSize
    nvid = th.stack(nvid)[...,sH:eH,sW:eW]
    vid = th.stack(vid)[...,sH:eH,sW:eW]
    # print(th.mean((nvid-vid)**2).item())
    return vid,nvid

def get_data_example(dcfg):
    root = Path("output/figures/crop_cat_chicken")
    device = "cuda:0"
    vid = vid_io.read_video(root).to(device)/255.
    nvid = vid + dcfg.sigma/255. * th.randn_like(vid)
    # print(vid.shape,nvid.shape)
    vid = vid[None,:].contiguous()
    nvid = nvid[None,:].contiguous()
    return vid,nvid

def get_center_boxes(ws,L,B):
    cH,cW = ws//2,ws//2
    sH,sW = cH-L//2,cW-L//2
    eH,eW = sH+L,sW+L
    boxes = th.Tensor([[[sH,sW,eH,eW]],[[sH,sW,eH,eW]]]).to("cuda").type(th.uint8)
    boxes = boxes.view(1,2,1,4).repeat(B,1,1,1)
    return boxes

def anno_flow(grid,boxes,color):
    B,wt,C,wh,ww = grid.shape
    grid = ((grid/grid.max())*255.).type(th.uint8)
    grid_annos = []
    for b in range(B):
        grid_annos_b = []
        for t in range(grid.shape[1]):
            grid_t = draw_bounding_boxes(grid[0,t],boxes[b,t],
                                     fill=False,colors=color)
            grid_annos_b.append(grid_t)
        grid_annos_b = th.stack(grid_annos_b)
        grid_annos.append(grid_annos_b)
    grid_annos = th.stack(grid_annos)
    return grid_annos

def get_peaks(dists,inds,K,L,boxes=True):

    B,wt,wh,ww = dists.shape
    dists = dists.view(B,wt,-1)
    twoOrthree = inds.shape[-1]
    inds = inds.view(B,wt,-1,twoOrthree)
    topk = th.topk(dists,K,dim=-1)
    topk_i = topk.indices[...,::5]

    inds_topk = []
    for i in range(twoOrthree):
        inds_topk_i = th.gather(inds[...,i],-1,topk_i)
        inds_topk.append(inds_topk_i)
    inds_topk = th.stack(inds_topk,-1)
    # print(inds_topk[0,0])
    # print(inds_topk.shape)

    # -- to boxes --
    if boxes == True:
        boxes = inds_topk[...,-2:]
        boxes = th.nn.functional.relu(boxes-L//2).flip(-1)
        boxes = th.cat([boxes,boxes+L],-1)
    else:
        boxes = inds_topk[...,-2:]

    return boxes

def get_grid(H,W,dtype,device):
    grid_y, grid_x = th.meshgrid(th.arange(0, H, dtype=dtype, device=device),
                                    th.arange(0, W, dtype=dtype, device=device))
    grid = th.stack((grid_y, grid_x), 2).float()  # W(x), H(y), 2
    grid = rearrange(grid,'H W two -> two H W')
    return grid

def show_groups(cfg,dcfg):


    # -- get video --
    set_seed(dcfg.seed)
    vid,nvid = get_data(dcfg)
    B,T,C,H,W = vid.shape

    # -- get sims --
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
    print(th.mean(flows.fflow**2).item(),th.mean(flows.bflow**2).item())
    # dists,inds = search(nvid,nvid,flows.fflow,flows.bflow)
    acc_flows = stnls.nn.accumulate_flow(flows.fflow,flows.bflow)
    th.cuda.synchronize()
    dists_0,inds_0 = search_p(nvid[:,1],nvid[:,2],flows.fflow[:,1])
    # inds_0 = th.cat([th.zeros_like(inds_0),inds_0],-1)
    dists_1,inds_1 = search_p(nvid[:,1],nvid[:,0],flows.bflow[:,1])
    # inds_1 = th.cat([2*th.ones_like(inds_1),inds_0],-1)
    dists = th.cat([dists_0,dists_1],-1)
    inds = th.cat([inds_0,inds_1],-2)
    # print("dists.shape: ",dists.shape,cfg.ws*cfg.ws)
    # print("inds.shape: ",inds.shape)

    # dists,inds = search_p.paired_vids(nvid,nvid,acc_flows,cfg.wt,skip_self=True)
    dists = th.exp(-10.*dists)
    del acc_flows
    stride1 = cfg.stride1

    dists = rearrange(dists,'b 1 (t h w) k -> b t h w k',h=H,w=W)
    inds = rearrange(inds,'b 1 (t h w) k tr -> b t h w k tr',h=H,w=W)
    dists = dists[:,0]
    inds = inds[:,0]
    B = dists.shape[0]

    # -- create offsets --
    offs = rearrange(inds,'b h w k tr -> (b k) tr h w')
    offs[:,-2:] = offs[:,-2:] - get_grid(H,W,th.float32,inds.device)[None,:]

    # -- across pixels --
    sH,sW = 100,100
    eH,eW = sH+100,sW+100
    dists = rearrange(dists[:,sH:eH,sW:eW],'b h w k -> (b h w) k')
    inds = rearrange(inds[:,sH:eH,sW:eW],'b h w k tr -> (b h w) k tr')
    offs = rearrange(offs[...,sH:eH,sW:eW],'(b k) tr h w -> (b h w) k tr',b=B)

    dists = rearrange(dists,'... (wt wh ww) -> ... wt wh ww',wh=cfg.ws,ww=cfg.ws)
    inds = rearrange(inds,'... (wt wh ww) tr -> ... wt wh ww tr',wh=cfg.ws,ww=cfg.ws)
    offs = rearrange(offs,'... (wt wh ww) tr -> ... wt wh ww tr',wh=cfg.ws,ww=cfg.ws)
    # print("dists.shape: ",dists.shape)
    for i in range(offs.shape[-4]):
        for j in range(2):
            offs[...,i,:,:,j] -= offs[...,i,:,:,j].mean(dim=(-1,-2),keepdim=True)
    offs[...,-2:] = (offs[...,-2:] + cfg.ws//2*stride1)/stride1
    # print(offs.min(),offs.max())
    offs = offs.int()

    # print("dists.shape: ",dists.shape)
    grid_annos = dists
    grid = th.zeros_like(grid_annos)
    L,K = 2,30
    boxes = get_peaks(dists,offs,K,L,False).long()
    # print((1.*boxes).mean((0,1,2)))

    del offs
    del dists
    del inds

    # print(boxes.min(),boxes.max(),boxes.shape)
    # print("grid.shape: ",grid.shape)
    # print("boxes.shape: ",boxes.shape)
    # print(boxes[0])
    B,wt,wh,ww = grid.shape
    B,wt,k,two = boxes.shape
    for wi in range(wt):
        for ki in range(k):
            for bi in range(B):
                grid[bi,wi,boxes[bi,wi,ki,0],boxes[bi,wi,ki,1]] += 1
    # print(grid.sum(),grid[0].sum(),grid[1].sum(),grid.sum()/(1.*grid[0].sum()))
    # for wi in range(wt):
    #     grid[:,wi] /= grid[:,wi].max()
    grid = th.mean(grid,dim=(0,1,))[None,None,]
    grid /= grid.max()
    prob = grid/grid.sum()
    # print(grid.shape)

    root = Path("output/figures/flow_error/")
    # vid_io.save_video(grid,root,"ave",itype="png")
    # print(prob[0,0,19:22,19:22])
    vid_io.save_video(nicer_image(grid),root,"ave",itype="png")

def run_exps(cfg,dcfg):


    # -- get video --
    set_seed(dcfg.seed)
    vid,nvid = get_data(dcfg)
    B,T,C,H,W = vid.shape

    # -- get sims --
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
    print(th.mean(flows.fflow**2).item(),th.mean(flows.bflow**2).item())
    # dists,inds = search(nvid,nvid,flows.fflow,flows.bflow)
    acc_flows = stnls.nn.accumulate_flow(flows.fflow,flows.bflow)
    th.cuda.synchronize()
    dists,inds = search_p.paired_vids(nvid,nvid,acc_flows,cfg.wt,skip_self=True)
    dists = th.exp(-10.*dists)
    dists /= th.sum(dists,-1,keepdim=True)
    del acc_flows
    stride1 = cfg.stride1

    dists = rearrange(dists,'b 1 (t h w) k -> b t h w k',h=H,w=W)
    inds = rearrange(inds,'b 1 (t h w) k tr -> b t h w k tr',h=H,w=W)
    dists = dists[:,1]
    inds = inds[:,1]
    B = dists.shape[0]

    # -- create offsets --
    offs = rearrange(inds,'b h w k tr -> (b k) tr h w')
    offs[:,1:] = offs[:,1:] - get_grid(H,W,th.float32,inds.device)[None,:]

    # -- across pixels --
    H,W = vid.shape[-2:]
    sH,sW = H//2+6,W//2+6
    eH,eW = sH+1,sW+1
    dists = rearrange(dists[:,sH:eH,sW:eW],'b h w k -> (b h w) k')
    inds = rearrange(inds[:,sH:eH,sW:eW],'b h w k tr -> (b h w) k tr')
    offs = rearrange(offs[...,sH:eH,sW:eW],'(b k) tr h w -> (b h w) k tr',b=B)
    print(dists.sum(-1))
    print(th.topk(dists[0],k=3))

    dists = rearrange(dists,'... (wt wh ww) -> ... wt wh ww',wh=cfg.ws,ww=cfg.ws)
    inds = rearrange(inds,'... (wt wh ww) tr -> ... wt wh ww tr',wh=cfg.ws,ww=cfg.ws)
    offs = rearrange(offs,'... (wt wh ww) tr -> ... wt wh ww tr',wh=cfg.ws,ww=cfg.ws)
    for i in range(offs.shape[-4]):
        for j in range(1,3):
            offs[...,i,:,:,j] -= offs[...,i,:,:,j].mean()
    offs[...,1:] = (offs[...,1:] + cfg.ws//2*stride1)/stride1
    offs = offs.int()
    print(offs[...,1:].min().item(),offs[...,1:].max().item())

    grid_annos = dists[:,:,None].repeat(1,1,3,1,1)
    L = 2
    K = 10
    B = grid_annos.shape[0]
    boxes = get_peaks(dists,offs,K,L)
    print("Pixels:")
    print((boxes-cfg.ws//2)*stride1)
    grid_annos = anno_flow(grid_annos,boxes,"blue")
    boxes = get_center_boxes(cfg.ws,L,B)
    grid_annos = anno_flow(grid_annos,boxes,"red")
    print("grid_annos.shape: ",grid_annos.shape)
    root = Path("output/figures/flow_error/")
    vid_io.save_video(nicer_image(grid_annos),root,"grid",itype="png")
    # vid_io.save_video(grid_annos,root,"grid",itype="png")

def nicer_image(img):
    print(img.shape)
    img = TF.resize(img[0],(128,128),InterpolationMode.NEAREST)[None,:]
    return img

def main():
    fstart = 0
    bs = 1
    nf = 3
    fend = fstart + nf - 1 + (bs-1)
    fn = "/home/gauenk/Documents/data/davis/DAVIS/ImageSets/2017/train-val.txt"
    vid_names = np.loadtxt(fn,str)
    vid_names = ["color-run"]
    vid_names = ["kid-football"]
    # vid_names = ["sunflower"]
    for vid_name in vid_names:
        # sH,sW,sSize = 192,300,64
        sH,sW,sSize = 18,8,170

        dcfg = edict({"dname":"davis","dset":"tr","vid_name":vid_name,"sigma":10.,
                      "nframes":nf,"frame_start":fstart,"frame_end":fend,
                      "isize":"512_512","seed":123,"sH":sH,"sW":sW,"sSize":sSize})
        ps = 3
        ws = 11
        s1 = 0.5
        cfgs = [edict({"name":"stnls","ps":7,"ws":41,"full_ws":False,
                       "wt":1,"k":-1,"stride0":1,"stride1":1.,"flow":True}),
                # edict({"name":"stnls","ps":ps,"ws":ws,"full_ws":False,
                #        "wt":1,"k":1,"stride0":1,"stride1":s1,"flow":True})
        ]
        for cfg in cfgs:
            results = run_exps(cfg,dcfg)
            # results = show_groups(cfg,dcfg)

if __name__ == "__main__":
    main()
