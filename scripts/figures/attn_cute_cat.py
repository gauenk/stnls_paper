"""

Show the attention mask for the cute cat.

"""

# -- misc --
import torch as th
import numpy as np
from PIL import Image

# -- read/write --
from pathlib import Path
from easydict import EasyDict as edict
from dev_basics.utils import vid_io

# -- space-time search --
import stnls
from dev_basics import flow


# -- viz --
from torchvision.utils import draw_segmentation_masks


def qnot_from_mask(mask,H,W,stride0):

    # -- get nums --
    nH = (H-1)//stride0+1
    nW = (W-1)//stride0+1
    nHW = nH*nW

    # -- rasterized --
    # mask = mask[...,::stride0,::stride0]
    args = th.where(mask[:,0]==0)
    argsT = args[0]
    argsH = th.div(args[1],stride0,rounding_mode='floor')
    argsW = th.div(args[2],stride0,rounding_mode='floor')
    qinds = argsT*nHW+argsH*nW+argsW
    return qinds


def q_from_mask(mask,H,W,stride0):

    # -- get nums --
    nH = (H-1)//stride0+1
    nW = (W-1)//stride0+1
    nHW = nH*nW

    # -- rasterized --
    args = th.where(mask[:,0]==1)
    argsT = args[0]
    argsH = th.div(args[1],stride0,rounding_mode='floor')
    argsW = th.div(args[2],stride0,rounding_mode='floor')
    qinds = argsT*nHW+argsH*nW+argsW
    # print(qinds)
    return qinds


def get_cat_mask(T,H,W):
    mask = th.zeros((T,1,H,W))*0.
    # mask[0,:,90:290,490+100:690+100] = 1
    D = 2

    # sH = 90//D
    # eH = 390//D
    # sW = 500//D
    # eW = sW+(300//D)

    sH = 150//D
    eH = sH+(300//D)
    sW = 150//D
    eW = sW+(300//D)

    # sH = 320
    # eH = 420
    # sW = 250
    # eW = 350

    # sH = 170
    # eH = sH+60
    # sW = 190
    # eW = sW+60

    mask[1,:,sH:eH,sW:eW] = 1
    return mask

def get_cat_mask_time(cat_mask,flows,ps,ws,wt,T,H,W):


    print(cat_mask.shape)
    args = th.where(cat_mask)
    argsT = args[0]
    argsH = args[2]
    argsW = args[3]
    # for i in range(len(argsT)):
    cat_mask = th.zeros_like(cat_mask)
    p2 = ps//2

    # argsT = [argsT[0]]
    # argsH = [th.mean(argsH*1.)]
    # argsW = [th.mean(argsW*1.)]
    # print(flows.fflow.shape)

    Q = len(argsT)
    for i in range(Q):
        # i = 0

        # -- mid --
        t = int(argsT[i].item())
        h,w = int(argsH[i].item()),int(argsW[i].item())
        cat_mask[t,:,h-ws//2-p2:h+ws//2+p2,w-ws//2-p2:w+ws//2+p2] = 1

        # -- fwd --
        for st in range(wt):
            if t+1 >= T: continue
            fflow = flows.fflow[0,t,:,h,w]
            h = min(max(int(h + fflow[1]+0.5),0),H)
            w = min(max(int(w + fflow[0]+0.5),0),W)
            cat_mask[t+1,:,h-ws//2-p2:h+ws//2+p2,w-ws//2-p2:w+ws//2+p2] = 1
            t+=1

        # -- bwd --
        t = int(argsT[i].item())
        h,w = int(argsH[i].item()),int(argsW[i].item())
        for st in range(wt):
            if t-1 < 0: continue
            bflow = flows.bflow[0,t,:,h,w]
            h = min(max(int(h + bflow[1]+0.5),0),H)
            w = min(max(int(w + bflow[0]+0.5),0),W)
            cat_mask[t-1,:,h-ws//2-p2:h+ws//2+p2,w-ws//2-p2:w+ws//2+p2] = 1
            t-=1

        # t = int(argsT[i].item())+1
        # # h,w = int(argsH[i].item()),int(argsW[i].item())
        # fflow = flows.fflow[0,t,:,h,w]
        # # h,w = int(argsH[i].item()),int(argsW[i].item())
        # # bflow = flows.bflow[0,t,:,h,w]
        # # if t-1 >= 0:
        # if t+1 < T:
        #     h = min(max(int(h + fflow[1]+0.5),0),H)
        #     w = min(max(int(w + fflow[0]+0.5),0),W)
        #     # print(t-1,h,w)
        #     cat_mask[t+1,:,h-ws//2:h+ws//2,w-ws//2:w+ws//2] = 1
    return cat_mask

def main():

    # -- params --
    device = "cuda:0"

    # -- read video --
    def read(fn): return th.tensor(np.array(Image.open(fn)).transpose(2,0,1))
    path = Path("figures/cute_cat/v2")
    fns = sorted([fn for  fn in path.iterdir()])
    vid = [read(fn) for fn in fns]
    vid = th.stack(vid)[None,]/255.
    vid = vid[...,::2,::2].contiguous()
    vid = vid.to(device)
    B,T,C,H,W = vid.shape
    print("vid.shape: ",vid.shape)

    # -- init non-local search --
    ws = 21
    wt = 1
    k = 50
    ps = 7
    stride0 = 4
    search = stnls.search.NonLocalSearch(ws,wt,ps,k,dist_type="l2",
                                         stride0=stride0,anchor_self=True)
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1

    # -- get flow --
    flows = flow.orun(vid,True,ftype="svnlb")
    # flows = flow.orun(vid,True,ftype="cv2_tvl1")

    # -- cat mask --
    ps_viz = 15
    cat_mask = get_cat_mask(T,H,W)
    cat_mask_time = get_cat_mask_time(cat_mask,flows,ps_viz,ws,wt,T,H,W)
    # print(cat_mask.sum())
    # cat_mask[1,:,-1,-1] = 1.
    # print(cat_mask.sum())
    vid_io.save_video(cat_mask,"output/figures/cute_cats/","cat_mask")
    vid_io.save_video(cat_mask_time,"output/figures/cute_cats/","cat_time")
    # exit(0)

    # -- run search --
    dists,inds = search(vid,vid,flows.fflow,flows.bflow,batchsize=1024)
    print("dists.shape,inds.shape: ",dists.shape,inds.shape)
    dists,inds = dists[0,0],inds[0,0]
    inds = inds.long()

    # -- create map --
    mask = th.zeros_like(vid[0,:,[0]])
    Q,K = dists.shape
    Qgrid = q_from_mask(cat_mask,H,W,stride0)
    print(Qgrid)
    Qnot_grid = qnot_from_mask(cat_mask,H,W,stride0)
    # Qgrid = [Qgrid[0]]
    # print(Qgrid)


    # patches = th.zeros((Q,K,1,C,ps,ps)).to(device)

    unfold = stnls.UnfoldK(15)
    patches = unfold(vid,inds[None,:].int())
    patches[:,Qnot_grid] = 0.
    print("per: ",patches.sum()/patches.numel())
    # dists[...] = 0.

    fold = stnls.FoldK(vid.shape)
    # print(dists.shape,inds.shape)
    mask,wmask = fold(patches,th.exp(-0.01*dists[None,:]),inds[None,:].int())
    print(mask.shape,mask.min(),mask.max(),wmask.min(),wmask.max())
    mask /= wmask
    mask = mask[0]
    print(mask.shape)
    for t in range(T):
        mask[t] /= mask[t].max()
    # mask = (mask > 0)*vid[0]
    # # mask = mask.ravel()
    # for q in Qgrid:
    #     for k in range(K):
    #         i,j = 0,0
    #         t = inds[q,k,0]
    #         h = inds[q,k,1]
    #         w = inds[q,k,2]
    #         dists_k = dists[q,k]
    #         mask[t,:,h,w] += th.exp(-dists_k)

    # mask = mask.view(T,1,H,W)
    # print(mask.min(),mask.max())
    print(th.where(mask>0))
    print(mask.min(),mask.max())
    # mask = th.log(mask+1e-10)
    # mask -= mask.min()
    # print(mask.min(),mask.max())
    mask /= mask.max()
    cat_mask = cat_mask[:,0].type(th.bool)
    cat_mask_time = cat_mask_time[:,0].type(th.bool)
    # cat_mask = (cat_mask*255.).type(th.uint8)
    mask = mask.cpu()
    mask = (mask*255.).type(th.uint8)
    # mask = mask[0]
    # print(mask.shape)
    # print(cat_mask.shape)
    # mask[1] = draw_segmentation_masks(mask[1], masks=cat_mask[1],
    #                                   alpha=0.3, colors="blue")
    # for t in range(T):
    #     mask[t] = draw_segmentation_masks(mask[t], masks=cat_mask_time[t],
    #                                       alpha=0.15, colors="yellow")

    vid_io.save_video(mask/255.,"output/figures/cute_cats/","mask")

    vid_io.save_video(mask[...,20:20+275,20:20+275]/255.,
                      "output/figures/cute_cats/","mask_cc")
    # vid_io.save_video(mask[...,:185,-205:-20]/255.,
    #                   "output/figures/cute_cats/","mask_cc")

    # -- temp --
    vid = vid[0].cpu()
    vid = (vid*255.).type(th.uint8)
    T = vid.shape[0]
    vid[1] = draw_segmentation_masks(vid[1], masks=cat_mask[1],
                                      alpha=0.5, colors="blue")
    for t in range(T):
        vid[t] = draw_segmentation_masks(vid[t], masks=cat_mask_time[t],
                                          alpha=0.3, colors="yellow")
    vid_io.save_video(vid,"output/figures/cute_cats/","vid")


if __name__ == "__main__":
    main()
