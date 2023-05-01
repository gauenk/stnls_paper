
import torch as th
import frame2frame
from frame2frame.nb2nb_loss import generate_mask_pair,generate_subimages
import stnls
import data_hub
from dev_basics import flow
from easydict import EasyDict as edict
from einops import rearrange

def run_exps(cfg,dcfg):

    # -- get video --
    data,loaders = data_hub.sets.load(dcfg)
    indices = data_hub.filter_subseq(data[dcfg.dset],dcfg.vid_name,
                                     dcfg.frame_start,dcfg.frame_end)
    # print(indices)
    # vid = data[dcfg.dset][indices[0]]['clean']
    device = "cuda:0"
    vid = data[dcfg.dset][0]['clean'].to(device)/255.
    # print(vid.numel())
    # vid = th.arange(vid.numel()).reshape(vid.shape).to(device)*1.
    # print(vid.shape)

    # -- get sims --
    if cfg.name == "nb2nb":
        mask0,mask1 = generate_mask_pair(vid)
        vid0 = generate_subimages(vid, mask0)
        vid1 = generate_subimages(vid, mask1)
        vid = vid[...,::2,::2].contiguous()
    elif cfg.name == "stnls":
        vid = vid[None,:].contiguous()
        search = stnls.search.NonLocalSearch(cfg.ws,cfg.wt,cfg.ps,cfg.k,
                                             nheads=1,dist_type="l2",
                                             stride0=cfg.stride0,
                                             anchor_self=True,use_adj=True)
        flows = flow.orun(vid,cfg.flow,ftype="svnlb")
        dists,inds = search(vid,vid,flows.fflow,flows.bflow)
        print("inds.shape: ",inds.shape)
        print(inds[0,0,0,0])
        print(inds[0,0,1,0])
        adj = 0#cfg.ps//2
        # adj = 0
        unfold = stnls.UnfoldK(cfg.ps,adj=adj,reflect_bounds=True)
        adj = 0#cfg.ps//2
        fold = stnls.iFoldz(vid.shape,adj=adj)
        inds = inds[:,0]
        print("inds.shape: ",inds.shape)
        patches0 = unfold(vid,inds[...,[4],:])
        patches1 = unfold(vid,inds[...,[5],:])
        T = vid.shape[0]
        print("-"*10 + " patches " +"-"*10)
        print(patches0[0,0,0,0])
        print("-"*10 + " video " +"-"*10)
        print(vid[0,0,:,:5,:5])
        print("patches0.shape: ",patches0.shape)
        print("vid.shape: ",vid.shape)
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
        vid0,wvid0 = fold(patches0)
        vid1,wvid1 = fold(patches1)

        # -- normalize --
        vid0 = vid0 / wvid0
        vid1 = vid1 / wvid1

        # -- view --
        print("-"*10 + " video0 " +"-"*10)
        print(vid0[0,:,:5,:5])


    # -- compute sims --
    loss0 = th.mean((vid - vid0)**2).item()
    loss1 = th.mean((vid - vid1)**2).item()
    print(loss0,loss1)

def main():
    dcfg = edict({"dname":"davis","dset":"val","vid_name":"snowboard","sigma":30,
                  "nframes":5,"frame_start":0,"frame_end":4,"isize":"128_128"})
    cfgs = [edict({"name":"nb2nb"}),
            edict({"name":"stnls","ps":3,"ws":15,
                   "wt":3,"k":10,"stride0":1,"flow":False})]
    for cfg in cfgs:
        results = run_exps(cfg,dcfg)

if __name__ == "__main__":
    main()
