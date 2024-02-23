
import torch as th
import numpy as np
from torchvision.transforms.functional import center_crop
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import save_image,make_grid

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
    dcfg.ntype = "g"
    data,loaders = data_hub.sets.load(dcfg)
    indices = data_hub.filter_subseq(data[dcfg.dset],dcfg.vid_name,
                                     dcfg.frame_start,dcfg.frame_end)
    print(indices)
    device = "cuda:0"
    nvid = [data[dcfg.dset][i]['noisy'].to(device)/255. for i in indices]
    vid = [data[dcfg.dset][i]['clean'].to(device)/255. for i in indices]
    nvid = th.stack(nvid)
    vid = th.stack(vid)
    print(th.mean((nvid-vid)**2).item())
    return vid,nvid

def get_data_example(dcfg):
    root = Path("output/figures/crop_cat_chicken")
    device = "cuda:0"
    vid = vid_io.read_video(root).to(device)/255.
    print(dcfg.sigma)
    nvid = vid + dcfg.sigma/255. * th.randn_like(vid)
    # print(vid.shape,nvid.shape)
    nvid = th.stack(nvid)
    vid = th.stack(vid)
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

def get_offsets(cfg,nvid,vid,flows):
    # -- get sims --
    B,T,C,H,W = vid.shape
    search = stnls.search.NonLocalSearch(cfg.ws,cfg.wt,cfg.ps,cfg.k,
                                         nheads=1,dist_type="l2",
                                         stride0=cfg.stride0,
                                         self_action='anchor',
                                         use_adj=False,
                                         full_ws=cfg.full_ws)
    search_p = stnls.search.PairedSearch(cfg.ws,cfg.ps,cfg.k,
                                         nheads=1,dist_type="l2",
                                         stride0=cfg.stride0,
                                         stride1=cfg.stride1,
                                         self_action=None,
                                         use_adj=False,
                                         full_ws=cfg.full_ws,
                                         # full_ws_time=cfg.full_ws,
                                         itype="float")
    stacking = stnls.agg.NonLocalGather(1,cfg.stride0,itype="float")
    # print(th.mean(flows.fflow**2).item(),th.mean(flows.bflow**2).item())
    # dists,inds = search(nvid,nvid,flows.fflow,flows.bflow)
    # acc_flows = stnls.nn.accumulate_flow(flows.fflow,flows.bflow)
    flows = stnls.nn.search_flow(flows.fflow,flows.bflow,cfg.wt,cfg.stride0)
    th.cuda.synchronize()
    dists,inds = search_p.paired_vids(nvid,nvid,flows,cfg.wt,skip_self=True)
    dists = th.exp(-10.*dists)
    dists /= th.sum(dists,-1,keepdim=True)
    # del acc_flows
    stride1 = cfg.stride1

    # print(dists.shape)
    dists = rearrange(dists,'b 1 t h w k -> b t h w k',h=H,w=W)
    inds = rearrange(inds,'b 1 t h w k tr -> b t h w k tr',h=H,w=W)
    # inds = rearrange(inds,'b t h w (wt wh ww) tr -> b t h w wt wh ww tr',
    #                  wh=cfg.ws,ww=cfg.ws)
    # print(inds[0,1,0,0])
    # print(inds[0,1,50:52,80:82,0,0])
    # print(inds[0,1,50:52,80:82,0,1:])
    # print(inds[0,1,50:52,80:82,1,0])
    # print(inds[0,1,50:52,80:82,1,1:])
    # print("-"*20)
    ki = cfg.ki
    dists = dists[0,1,:,:,ki]
    inds = inds[0,1,...,ki,1:]
    B = dists.shape[0]

    # -- create offsets --
    offs = rearrange(inds,'h w tw -> tw h w')
    # offs = offs-get_grid(H,W,th.float32,inds.device)
    offs = rearrange(offs,'tr h w -> h w tr')
    return offs

def plot_grid(ax,X,Y,col,s):
    for hi in range(X.shape[0]):
        for wi in range(X.shape[1]):
            ax.scatter(X[hi,wi], Y[hi,wi],color=col,s=s)

def im_plot(dcfg,vid):
    fig,ax=plt.subplots(1,1,
                        figsize=(3,3),
                        dpi=dcfg.dpi,
                        tight_layout=True)
    ax.set_position([0, 0, 1, 1]) # Critical!
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    vid = rearrange(vid.cpu(),'c h w -> h w c')
    the_image = ax.imshow(
        vid,zorder=0,alpha=1.0,
        origin="upper",
        interpolation="nearest",
    )
    return fig,ax

def get_plt_image(fig,ax):
    ax.axis("off")
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
    #                     hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout(pad=0)
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    canvas = fig.canvas
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8')\
            .reshape(int(height), int(width), 3)
    plt.close("all")
    img = th.from_numpy(img)
    img = rearrange(img,'h w c -> c h w')
    return img

def run_exps(cfg,dcfg):


    # -- get video --
    set_seed(dcfg.seed)
    vid,nvid = get_data(dcfg)
    flows = flow.orun(nvid,cfg.flow,ftype="cv2")

    # -- crop here --
    sH,sW,sSize = dcfg.sH,dcfg.sW,dcfg.sSize
    eH,eW = sH+sSize,sW+sSize
    nvid = nvid[...,sH:eH,sW:eW]
    vid = vid[...,sH:eH,sW:eW]
    flows.fflow = flows.fflow[...,sH:eH,sW:eW]
    flows.bflow = flows.bflow[...,sH:eH,sW:eW]
    B,T,C,H,W = vid.shape
    print(vid.shape)


    # -- get sims --
    cfg.ki = dcfg.ki
    off_norm = get_offsets(cfg,nvid,vid,flows)
    cfg.ws = 1
    off_flow = get_offsets(cfg,nvid,vid,flows)
    cfg.ws = 51
    cfg.stride1 = 1
    off_big = get_offsets(cfg,nvid,vid,flows)

    # -- across pixels --
    # zstrip = th.zeros_like(vid[0,0,:,:,:2])
    # stack = th.cat([vid[0,1],zstrip,vid[0,0]],-1).cpu()
    # stack = rearrange(stack,'c h w -> h w c')

    # plt.colorbar(the_image)
    cols = ['red','orange','b',]#,'b']
    off_list = [off_big,off_flow]
    s = 40.
    skip = 2
    sH,sW,sSize = dcfg.ssH,dcfg.ssW,dcfg.ssSize
    # sH,sW,sSize = 22,25,10
    # sH,sW,sSize = 36,20,10
    # sH,sW,sSize = 50,85,10
    eH,eW = sH+sSize,sW+sSize
    shiftW = W + 2 # zpad
    # print(sH,eH,sW,eW,H,W)
    H,W = vid.shape[-2:]
    Y, X = np.mgrid[sH:eH:skip,sW:eW:skip]
    Y = th.from_numpy(Y)
    X = th.from_numpy(X)
    dX_big = off_norm[sH:eH:skip,sW:eW:skip,1].cpu()
    dY_big = off_norm[sH:eH:skip,sW:eW:skip,0].cpu()
    dX_flow = off_flow[sH:eH:skip,sW:eW:skip,1].cpu()
    dY_flow = off_flow[sH:eH:skip,sW:eW:skip,0].cpu()

    # -- frames --
    ti = 1
    tj = ti+1 if cfg.ki == 0 else ti-1

    # -- plot reference --
    fig,ax = im_plot(dcfg,vid[0,ti])
    col = cols[0]
    A = X
    B = Y
    plot_grid(ax,A,B,col,s)
    ref = get_plt_image(fig,ax)

    # -- plot flow + correction --
    fig,ax = im_plot(dcfg,vid[0,tj])
    col = cols[1]
    A = X+dX_flow
    B = Y+dY_flow
    plot_grid(ax,A,B,col,s)


    col = cols[2]
    A = X+dX_big
    B = Y+dY_big
    plot_grid(ax,A,B,col,s)

    adj = get_plt_image(fig,ax)

    return nicer_image(ref),nicer_image(adj)


def nicer_image(img):
    img = TF.resize(img,(256,256),InterpolationMode.NEAREST)
    return img

def main():
    fstart = 0
    bs = 1
    nf = 3
    fend = fstart + nf - 1 + (bs-1)
    fn = "/home/gauenk/Documents/data/davis/DAVIS/ImageSets/2017/train-val.txt"
    vid_names = np.loadtxt(fn,str)
    vid_names = [#"color-run:1:215-308-32:13-21-10",
        # "walking:1:192-128-32:12-18-10",
        # "scooter-board:1:112-256-220-362",
        # "scooter-gray:1:64-192-160-288",
        # "swing:0:96-292-192-388",
        #
        "scooter-board:1:130-270-48:20-30-10",
        "stroller:0:145-94-48:15-25-10",
        "hockey:0:60-230-48:30-30-10",
        "scooter-gray:1:84-178-48:10-36-10",
        #
        #
        # "swing:0:100-200-128:10-10-10",
        # "drone:0:230-200-48:12-18-10",
        # "color-run:1:215-308-32:13-21-10",
        # "kid-football:0:50-90-48:8-8-10",
        #
                 # "tuk-tuk:1:64-240-64:0-0-10"
                 ]
    vid = []
    # vid_names = ["tennis:207-303-40"]
    for vid_info in vid_names:
        # sH,sW,sSize = 207,303,40
        info = vid_info.split(":")
        vid_name = info[0]
        ki = int(info[1])
        sH,sW,sSize = [int(x) for x in info[2].split("-")]
        ssH,ssW,ssSize = [int(x) for x in info[3].split("-")]
        dcfg = edict({"dname":"davis","dset":"tr","vid_name":vid_name,"sigma":15.,
                      "nframes":nf,"frame_start":fstart,"frame_end":fend,
                      "isize":"512_512","seed":123,"sH":sH,"sW":sW,"sSize":sSize,
                      "ssH":ssH,"ssW":ssW,"ssSize":ssSize,"dpi":100,"ki":ki})
        ps = 7
        ws = 9
        s1 = 1.
        cfgs = [# edict({"name":"stnls","ps":3,"ws":41,"full_ws":False,
                #        "wt":1,"k":1,"stride0":1,"stride1":1.,"flow":True}),
                edict({"name":"stnls","ps":ps,"ws":ws,"full_ws":False,
                       "wt":1,"k":1,"stride0":1,"stride1":s1,"flow":True})
        ]
        for cfg in cfgs:
            ref,adj = run_exps(cfg,dcfg)
            # results = show_groups(cfg,dcfg)
            vid.append(th.stack([ref,adj]))
    vid = rearrange(th.stack(vid),'b two c h w -> (two b) c h w')
    print(vid.shape)
    grid = make_grid(vid,nrow=vid.shape[0]//2,pad_value=0.,padding=1)
    grid = grid/255.
    save_image(grid,"flow_error_v2.png")

if __name__ == "__main__":
    main()
