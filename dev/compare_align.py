
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
    return vid,nvid

def run_natten(nvid,acc_flows,k,ws,wt):
    def forward(q,k,flow_unused):
        attn = natten2dqkrpb(q,k,None,ws,1)
        attn = attn.softmax(dim=-1)
        topk = th.topk(attn,k,decreasing=True)
        print(topk.indices)
        topk.values,topk.indices
    dists,inds = paired_vids(forward,nvid,nvid,wt,skip_self=True)
    # attn = natten2dqkrpb(nvid, None, ksize, 1)
    # attn = attn.softmax(dim=-1)
    # topk = th.topk(attn,k)
    # x = natten2dav(attn, v, dilation)
    return topk.values,topk.indices

def run_stnls(search_p,nvid,acc_flows,wt):
    dists,inds = search_p.paired_vids(nvid,nvid,acc_flows,wt,skip_self=True)
    return dists,inds

def run_exps(cfg,dcfg):

    # -- init --
    timer = ExpTimer()
    memer = GpuMemer()

    # -- get video --
    set_seed(dcfg.seed)
    vid,nvid = get_data(dcfg)
    vid = vid[None,:].contiguous()
    nvid = nvid[None,:].contiguous()
    # print("vid.shape: ",vid.shape)
    # print(vid.max(),nvid.max())
    # psnrs = compute_psnrs(vid,nvid,div=1.)
    # print("psnrs: ",psnrs)

    # -- get sims --
    # print(cfg.ws,cfg.ps,cfg.k,cfg.ps_stack,cfg.stride0)
    search = stnls.search.NonLocalSearch(cfg.ws,cfg.wt,cfg.ps,cfg.k,
                                         nheads=1,dist_type="l2",
                                         stride0=cfg.stride0,
                                         self_action="remove",
                                         use_adj=False,full_ws=cfg.full_ws)
    search_p = stnls.search.PairedSearch(cfg.ws,cfg.ps,cfg.k,
                                         nheads=1,dist_type="l2",
                                         stride0=cfg.stride0,
                                         stride1=cfg.stride1,
                                         self_action=None,use_adj=False,
                                         full_ws=cfg.full_ws,itype="float")
    # print("cfg.k: ",cfg.k)
    # print("vid.shape: ",vid.shape)
    # print(vid[0,:,0,:2,:2])
    stacking = stnls.agg.NonLocalGather(cfg.ps_stack,cfg.stride0,itype="float")
    with TimeIt(timer,'flow'):
        with MemIt(memer,'flow'):
            flows = flow.orun(nvid,cfg.flow,ftype="cv2")
    flow_norm = (flows.fflow.abs().mean() + flows.bflow.abs().mean()).item()/2.
    # acc_flows = stnls.nn.accumulate_flow(flows.fflow,flows.bflow)
    acc_flows = stnls.nn.search_flow(flows.fflow,flows.bflow,cfg.wt,cfg.stride0)
    # print(acc_flows.shape)
    # print(acc_flows)
    with TimeIt(timer,'search'):
        with MemIt(memer,'search'):
            # if stype == "nat":
            #     dists,inds = run_natten(nvid,acc_flows,cfg.k,cfg.ws,cfg.wt)
            # else:
            dists,inds = run_stnls(search_p,nvid,acc_flows,cfg.wt)
            # dists,inds = search_p.paired_vids(nvid,nvid,acc_flows,cfg.wt,skip_self=True)
    ones = th.ones_like(dists)
    # print(inds[0,0,1])
    stack = stacking(vid,ones,inds)[:,0]
    # print("stack.shape: ",stack.shape)

    # -- compute sims --
    vid = vid[:,None].repeat(1,2,1,1,1,1)
    vid = rearrange(vid,'b k t c h w -> b (k t) c h w')
    stack = rearrange(stack,'b k t c h w -> b (k t) c h w')
    psnrs = compute_psnrs(stack,vid)
    print(psnrs)
    # psnrs = compute_psnrs(stack,vid).mean().item()
    alloc = memer['search']['alloc']
    res = memer['search']['res']

    return psnrs,flow_norm,timer['flow'],timer['search'],alloc,res

def run_it():

    # fn = "/home/gauenk/Documents/data/set8/image_sets/all.txt"
    # dname,dset = "set8","te"
    fn = "/home/gauenk/Documents/data/davis/DAVIS/ImageSets/2017/train.txt"
    dname,dset = "davis","tr"
    vid_names = np.loadtxt(fn,str)
    # tough ones; dance-jump, dancing, dog-agility
    # vid_names = ["cat-girl","classic-car","color-run","dog-gooses","drone","hockey","horsejump-low","kid-football","lady-running","lindy-hop","lucia","motorcross-bumps","motorbike","paragliding","scooter-board","scooter-grey","skate-park","snowboard","stroller","stunt","surf","swing","tennis","tractor-sand","tuk-tuk","upside-down","walking"]
    # vid_names = [vid_names[3],vid_names[-1]]
    info = {"psnrs":[],"flow_norm":[],"ws":[],"flow":[],"vid_name":[],
            "ftime":[],"time":[],"res":[],"alloc":[]}
    # psnrs = []
    # flow_norms = []
    for vid_name in vid_names:
        print(vid_name)
        fstart = 0
        fend = fstart + 10 - 1
        dcfg = edict({"dname":dname,"dset":dset,"vid_name":vid_name,"sigma":15.,
                      "nframes":5,"frame_start":fstart,"frame_end":fend,
                      "isize":None,"seed":123})
        ps = 3
        ps_stack = 3
        ws = 11
        s0 = 1
        s1 = 1
        # ws_grid = [1,3,9,15,21,27,33]
        ws_grid = [1,3,11,15,21,27,33]
        cfgs = []
        # for ws in ws_grid:
        #     # if ws <= 13: ps_i = ws
        #     # else: ps_i = ps
        #     ps_i = ps
        #     cfgs.append(edict({"name":"stnls","ps":ps_i,"ps_stack":ps_stack,
        #                        "ws":ws,"full_ws":False,
        #                        "wt":1,"k":1,"stride0":s0,"stride1":s1,"flow":False}))
        for ws in ws_grid:
            cfgs.append(edict({"name":"stnls","ps":ps,"ps_stack":ps_stack,
                               "ws":ws,"full_ws":False,
                               "wt":1,"k":1,"stride0":s0,"stride1":s1,"flow":True}))
        # cfgs = [edict({"name":"stnls","ps":ps,"ps_stack":ps_stack,
        #                "ws":ws,"full_ws":False,
        #                "wt":1,"k":1,"stride0":1,"stride1":s1,"flow":False}),
        #         edict({"name":"stnls","ps":1,"ps_stack":1,
        #                "ws":1,"full_ws":False,
        #                "wt":1,"k":3,"stride0":1,"stride1":.1,"flow":True}),
        #         edict({"name":"stnls","ps":ps,"ps_stack":ps_stack,
        #                "ws":ws,"full_ws":False,
        #                "wt":1,"k":1,"stride0":s0,"stride1":s1,"flow":True}),
        #         edict({"name":"stnls","ps":ps,"ps_stack":ps_stack,
        #                "ws":33,"full_ws":False,
        #                "wt":1,"k":1,"stride0":s0,"stride1":s1,"flow":True})
        # ]
        for cfg in cfgs:
            print("ws: ",cfg.ws)
            psnr,flow_norm,ftime,time,alloc,res = run_exps(cfg,dcfg)
            info['psnrs'].append(psnr)
            info['flow_norm'].append(flow_norm)
            info['ws'].append(cfg.ws)
            info['flow'].append(cfg.flow)
            info['vid_name'].append(vid_name)
            info['ftime'].append(ftime)
            info['time'].append(time)
            info['alloc'].append(alloc)
            info['res'].append(res)
    return info

def fill_natten_stats(df):
    info = {"ws":[1,3,9],"time":[.09070,.10070,.11070],
            "res":[10.,11.,12.],"alloc":[7,8,9]}
    inds = np.logical_and(df['ws'] <= 13,df['flow'] == False)
    # df = df[inds]
    for idx in range(len(info['ws'])):
        inds_b = np.logical_and(df['ws'] == info['ws'][idx],df['flow'] == False)
        inds = np.where(inds_b)[0]
        for jdx in inds:
            for key in info.keys():
                df.loc[jdx,key] = info[key][idx]

def compare_align_quality(df):
    # -- summary --
    fields = ['psnrs','flow_norm','time','ftime','res','alloc','flow','ws']
    summ = df[fields].groupby(['flow','ws']).agg("mean").reset_index(drop=False)
    summ = summ.sort_values(["flow","ws"])
    # summ['time'] = summ['time'] + summ['ftime']

    # -- plot --
    dpi = 200
    ginfo = {'width_ratios': [1.,],'wspace':0, 'hspace':0.0,
             "top":0.90,"bottom":0.13,"left":.15,"right":0.95}
    fig,ax = plt.subplots(figsize=(5,3.5),gridspec_kw=ginfo,dpi=200)

    alpha = 0.3
    summ_f = summ[summ['flow'] == True]
    ax.scatter(summ_f['time'],summ_f['psnrs'],s=50*summ_f['res'],color='b',alpha=alpha)
    ax.plot(summ_f['time'],summ_f['psnrs'],'b->',label="Shifted-NLS")
    summ_f = summ[summ['flow'] == False]
    ax.scatter(summ_f['time'],summ_f['psnrs'],s=50*summ_f['res'],color='r',alpha=alpha)
    ax.plot(summ_f['time'],summ_f['psnrs'],'r->',label="NLS")

    ymax,ymin = summ['psnrs'].max(),summ['psnrs'].min()
    ygrid = np.linspace(ymin,ymax,5)
    ax.set_yticks(ygrid)
    ax.set_yticklabels(["%2.1f" % y for y in ygrid])

    # ax.set_xscale('log')
    xmax,xmin = summ['time'].max(),summ['time'].min()
    xgrid = np.linspace(xmin,xmax,5)
    ax.set_xticks(xgrid)
    ax.set_xticklabels(["%1.2f" % x for x in xgrid])

    ax.set_xlabel("Runtime (seconds)",fontsize=13)
    ax.set_ylabel("Aligned Quality (dB)",fontsize=13)
    ax.set_title("Aligning Adjacent Frames",fontsize=15)

    ax.legend(framealpha=0.0,title="Search Method")

    plt.savefig("align_quality.png",transparent=True)
    plt.close("all")

def compare_align_motion(df):
    # -- summary --
    fields = ['psnrs','flow_norm','time','ftime','res','alloc','flow','ws']
    df = df[df['ws'] == 9]
    summ = df
    # summ = df[fields].groupby(['flow','ws']).agg("mean").reset_index(drop=False)
    # summ = summ.sort_values(["flow","ws"])
    # summ['time'] = summ['time'] + summ['ftime']
    flow_norm = summ[summ['flow'] == True]['flow_norm']
    yesflow_psnrs = summ[summ['flow'] == True]['psnrs']
    yesflow_ws = summ[summ['flow'] == True]['ws']/summ['ws'].max()
    noflow_psnrs = summ[summ['flow'] == False]['psnrs']
    noflow_ws = summ[summ['flow'] == False]['ws']/summ['ws'].max()
    delta_psnr = yesflow_psnrs.to_numpy() - noflow_psnrs.to_numpy()

    args = np.argsort(delta_psnr)
    print(delta_psnr[args])
    print(yesflow_psnrs.to_numpy()[args])
    print(noflow_psnrs.to_numpy()[args])
    # print(yesflow_psnrs.shape)
    # print(delta_psnr.shape)
    # print(flow_norm.shape)

    # -- plot --
    dpi = 200
    ginfo = {'width_ratios': [1.,],'wspace':0, 'hspace':0.0,
             "top":0.90,"bottom":0.14,"left":.15,"right":0.95}
    fig,ax = plt.subplots(figsize=(5,3.5),gridspec_kw=ginfo,dpi=200)

    # summ_f = summ[summ['flow'] == True]
    scale = 10
    ax.scatter(flow_norm,delta_psnr,s=scale,color='k',
               alpha=1.,label="Shifted-NLS")
    # ax.scatter(flow_norm,noflow_psnrs,s=scale*noflow_ws,color='r',
    #            alpha=1.,label="NLS")
    # ax.plot(summ_f['time'],summ_f['psnrs'],'b->',label="Shifted-NLS")
    # summ_f = summ[summ['flow'] == False]
    # ax.scatter(summ_f['time'],summ_f['psnrs'],s=50*summ_f['res'],color='r',alpha=0.5)
    # ax.plot(summ_f['time'],summ_f['psnrs'],'r->',label="NLS")

    ax.axhline(0.,linestyle='--',color='k')

    # ymax,ymin = summ['psnrs'].max(),summ['psnrs'].min()
    ymax,ymin = delta_psnr.max(),delta_psnr.min()
    ygrid = np.linspace(ymin,ymax,5)
    ax.set_yticks(ygrid)
    ax.set_yticklabels(["%2.1f" % y for y in ygrid])

    # ax.set_xscale('log')
    xmax,xmin = flow_norm.min(),flow_norm.max()
    xgrid = np.linspace(xmin,xmax,5)
    ax.set_xticks(xgrid)
    ax.set_xticklabels(["%1.2f" % x for x in xgrid])

    ax.set_xlabel("Average Flow (pixels)",fontsize=13)
    ax.set_ylabel("Aligned Quality Difference (dB)",fontsize=13)
    ax.set_title("Impact of Motion",fontsize=15)

    # ax.legend(framealpha=0.0,title="Search Method")

    plt.savefig("align_motion.png",transparent=True)
    plt.close("all")


def main():
    cache = cache_io.FileCache(".cache_io/alignment_s15/")
    # cache = cache_io.FileCache(".cache_io/alignment/")
    overwrite = False
    info = cache.read("compare_align")
    if info is None or overwrite:
        info = run_it()
        cache.write("compare_align",info,overwrite=overwrite)
    df = pd.DataFrame(info)
    print(df)

    # -- summary --
    # fields = ['psnrs','flow_norm','time','ftime','res','alloc','flow','ws']
    # summ = df[fields].groupby(['flow','ws']).agg("mean").reset_index(drop=False)
    # summ = summ.sort_values(["flow","ws"])
    # summ['time'] = summ['time'] + summ['ftime']

    # -- plot --
    compare_align_quality(df)
    compare_align_motion(df)


if __name__ == "__main__":
    main()

