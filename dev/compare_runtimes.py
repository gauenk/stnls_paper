"""

Check runtimes & memory

"""

# -- basic --
import torch as th
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- plot --
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# -- caching --
import cache_io

# -- stnls --
import stnls

# -- natten --
from natten import NeighborhoodAttention2D
from natten.functional import natten2dav, natten2dqkrpb

# -- bench --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from dev_basics import flow


def run_natten(q,k,v,K,ksize=3,rpb=None,dilation=1):
    attn = natten2dqkrpb(q, k, None, ksize, dilation)
    attn = attn.softmax(dim=-1)
    topk = th.topk(attn,K,-1,largest=True)
    dists,inds = topk.values,topk.indices
    return attn

def run_stnls(q,k,v,zflow,ws,wt,ps,K,stride0,HD):
    search = stnls.search.NonLocalSearch(ws,wt,ps,K,dist_type="prod",
                                         nheads=HD,stride0=stride0)
    dists,inds = search(q,k,zflow,zflow)
    return dists,inds

def run_n3net(q,k,v,zflow,ws,wt,ps,K,stride0,HD):
    search = stnls.search.N3MatMultSearch(ws,wt,ps,K,dist_type="prod",
                                          nheads=HD,stride0=stride0)
    dists,inds = search(q,k,zflow,zflow)
    return dists,inds

def run_conv2d(q,k,v,ksize):
    F = q.shape[-3]
    weight = th.zeros((F,F,ksize,ksize)).to(q.device)
    q = q.flatten(0,2)
    out = th.nn.functional.conv2d(q[0],weight)
    return out

def get_nat_data(B,HD,T,F,H,W,St):
    F_HD = F//HD
    assert F_HD*HD == F
    q = th.randn((B*T*St,H,W,1,HD,F_HD),device="cuda").permute(3, 0, 4, 1, 2, 5)[0]
    k = th.randn((B*T*St,H,W,1,HD,F_HD),device="cuda").permute(3, 0, 4, 1, 2, 5)[0]
    v = th.randn((B*T*St,H,W,1,HD,F_HD),device="cuda").permute(3, 0, 4, 1, 2, 5)[0]
    return q,k,v

def get_stnls_data(B,HD,T,F,H,W):
    q = th.randn((B,T,F,H,W),device="cuda")
    k = th.randn((B,T,F,H,W),device="cuda")
    v = th.randn((B,T,F,H,W),device="cuda")
    return q,k,v

def exec_method(cfg):
    # -- init --
    timer = ExpTimer()
    memer = GpuMemer()

    B,HD,T,F,H,W = cfg.B,cfg.HD,cfg.T,cfg.F,cfg.H,cfg.W
    ws,wt,ps,K,stride0 = cfg.ws,cfg.wt,cfg.ps,cfg.K,cfg.stride0

    if cfg.name == "n3net":
        q,k,v = get_stnls_data(B,HD,T,F,H,W)
        zflow = th.zeros((B,T,2,H,W),device="cuda")
        print("[n3net] q.shape: ",q.shape)
        with TimeIt(timer,"search"):
            with MemIt(memer,"search"):
                run_n3net(q,k,v,zflow,ws,wt,ps,K,stride0,HD)
    elif cfg.name == "stnls":
        q,k,v = get_stnls_data(B,HD,T,F,H,W)
        zflow = th.zeros((B,T,2,H,W),device="cuda")
        print("[stnls] q.shape: ",q.shape)
        with TimeIt(timer,"search"):
            with MemIt(memer,"search"):
                run_stnls(q,k,v,zflow,ws,wt,ps,K,stride0,HD)
    elif cfg.name == "nat":
        St = 2*wt+1
        q,k,v = get_nat_data(B,HD,T,F,H,W,St)
        print("[nat] q.shape: ",q.shape)
        with TimeIt(timer,"search"):
            with MemIt(memer,"search"):
                run_natten(q,k,v,K,ws)

    results = {"time":timer['search'],
               "res":memer['search']['res'],
               "alloc":memer['search']['alloc']}
    return results


def compare_memory(df):

    # -- unpack --
    fields = ["time","res","alloc","ws","stride0"]
    nat = df[df['name'] == 'nat']
    nat = nat[fields].groupby(["ws"]).agg("mean").reset_index()
    print(nat)
    snls = df[df['name'] == 'stnls'][fields].groupby(["ws","stride0"]).agg("mean").reset_index()
    print(snls)
    n3net = df[df['name'] == 'n3net'][fields].groupby(["ws","stride0"]).agg("mean").reset_index()
    print(n3net)
    mems = []


    # -- init --
    dpi = 200
    ginfo = {'width_ratios': [1.,],'wspace':0, 'hspace':0.0,
             "top":0.92,"bottom":0.16,"left":.13,"right":0.98}
             # "top":0.92,"bottom":0.14,"left":.11,"right":0.98}
    fig,ax = plt.subplots(figsize=(4.5,2.81),gridspec_kw=ginfo,dpi=200)

    # -- plot --
    ax.plot(nat['ws'],nat['res'],label="NATTEN (S=1)",color='k')
    mems.append(nat['res'].to_numpy())
    for name,gdf0 in df.groupby("name"):
        if name == "nat": continue
        nlabel = "Shifted-NLS" if name == "stnls" else "N3Net"
        color = 'b' if name == "stnls" else "r"
        ls,lidx = ['-','--'],0
        for (stride0,ps),gdf1 in gdf0.groupby(["stride0","ps"]):
            # if name == "n3net" and stride0 == 1:
            #     lidx += 1
            #     continue
            # label = nlabel + " (P=%d)" % (ps)
            label = nlabel + " (S=%d)" % (stride0)
            ax.plot(gdf1['ws'],gdf1['res'],label=label,color=color,linestyle=ls[lidx])
            mems.append(gdf1['res'].to_numpy())
            lidx += 1

    # -- format --
    ax.legend(framealpha=0.,ncol=2)
    ax.set_title("Forward Pass Memory")

    wgrid = nat['ws'].to_numpy()
    ax.set_xticks(wgrid)
    ax.set_xticklabels(["%d" % w for w in wgrid])

    ax.set_ylabel("GPU Memory (GB)")
    ax.set_xlabel("Spatial Window Size ($W_s$)")

    mems = np.concatenate(mems)
    ymin,ymax = mems.min(),mems.max()
    # ax.set_yticks(ygrid)
    # ax.set_yticklabels(["%1.2f" % (w) for w in ygrid])
    # ax.set_ylim([0,ymax*1.35])
    ax.set_yscale("log")
    ax.set_ylim([0,ymax*5])

    # -- save --
    plt.savefig("search_memory.png",transparent=True)
    plt.close("all")


def compare_times(df):

    # -- unpack --
    print("-"*5 + " times " + "-"*5)
    fields = ["time","res","alloc","F","stride0"]
    nat = df[df['name'] == 'nat']
    nat = nat[fields].groupby(["F"]).agg("mean").reset_index()
    print(nat)
    snls = df[df['name'] == 'stnls'][fields].groupby(["F","stride0"]).agg("mean").reset_index()
    print(snls)
    n3net = df[df['name'] == 'n3net'][fields].groupby(["F","stride0"]).agg("mean").reset_index()
    print(n3net)
    times = []


    # -- init --
    dpi = 200
    ginfo = {'width_ratios': [1.,],'wspace':0, 'hspace':0.0,
             "top":0.92,"bottom":0.16,"left":.15,"right":0.98}
    fig,ax = plt.subplots(figsize=(4.5,2.81),gridspec_kw=ginfo,dpi=200)

    # -- plot --
    ax.plot(nat['F'],nat['time'],label="NATTEN (S=1)",color='k')
    times.append(nat['time'].to_numpy())
    for name,gdf0 in df.groupby("name"):
        if name == "nat": continue
        nlabel = "Shifted-NLS" if name == "stnls" else "N3Net"
        color = 'b' if name == "stnls" else "r"
        ls,lidx = ['-','--'],0
        for (stride0,ps),gdf1 in gdf0.groupby(["stride0","ps"]):
            if name == "n3net" and stride0 == 1:
                lidx += 1
                continue
            # label = nlabel + " (P=%d)" % (ps)
            label = nlabel + " (S=%d)" % (stride0)
            ax.plot(gdf1['F'],gdf1['time'],label=label,color=color,linestyle=ls[lidx])
            times.append(gdf1['time'].to_numpy())
            lidx += 1

    # -- format --
    ax.legend(framealpha=0.,ncol=2)
    ax.set_title("Forward Pass Runtimes")

    nftrs = nat['F'].to_numpy()
    ax.set_xticks(nftrs)
    ax.set_xticklabels(["%d" % (w//4) for w in nftrs])

    times = np.concatenate(times)
    ymin,ymax = times.min(),times.max()
    ygrid = np.linspace(0,ymax,5)
    ygrid = np.r_[ygrid,2*ygrid[-1]-ygrid[-2]]
    ax.set_yticks(ygrid)
    ax.set_yticklabels(["%1.2f" % (w) for w in ygrid])
    ax.set_ylim([0,ymax*1.35])

    ax.set_ylabel("Runtime (seconds)")
    ax.set_xlabel("Number of Features per Head")

    # -- save --
    plt.savefig("search_runtimes.png",transparent=True)
    plt.close("all")


def fine_grid(cfg):
    exps_cfg = {"cfg":cfg,
                "group0":{"name":["stnls","n3net","nat"]},
                "group1":{"ws":[3,5,9,7,13]},
                "group2":{"F":[3,16,16,16,32,32,32],
                          "HD":[1,1,4,8,1,4,8]},
                "group3":{"stride0":[1,1,2,2,4],
                           "ps":[3,7,3,7,7]},
    }
    exps = cache_io.exps.unpack(exps_cfg)
    exps_cfg = {"cfg":cfg,
                "group0":{"name":["stnls","n3net"]},
                "group1":{"F":[3,16,16,16,32,32,32],
                          "HD":[1,1,4,8,1,4,8]},
                "group2":{"ws":[3,5,9,7,13]},
                "group3":{"stride0":[1,2,4]},
                "group4":{"ps":[10,7]},
    }
    exps += cache_io.exps.unpack(exps_cfg)
    return exps

def selected_grid(cfg):
    # SELECT = {"F":[32,3,3,3,3,3,3,32,32,32,32,32,32],
    #           "HD":[1,1,1,1,1,1,1,4,4,4,4,4,4],
    #           "stride0":[1,4,1,1,1,2,2,2,1,1,1,2,2,2],
    #           "ps":[7,3,3,7,3,3,3,3,3,3,3,3,3],
    #           "ws":[13,3,9,9,3,9,13,3,9,13,3,9,13],}
    SELECT = {"group0":{"F":[32,64,128,256]},
              "group1":{"HD":[1,2,4]},
              "group2":{"stride0":[1,2]},
              "group3":{"ps":[1,3]},
              "group4":{"ws":[3,5,7,9,11,13]}}
    exps_cfg = {"cfg":cfg,
                "group10":{"name":["stnls","n3net","nat"]}}
    exps_cfg = exps_cfg | SELECT
    exps = cache_io.exps.unpack(exps_cfg)
    return exps

def main():

    # -- config --
    # cfg = edict({"B":1,"T":5,"H":224,"W":224,"K":3,"wt":1})
    # cfg = edict({"B":1,"T":5,"H":192,"W":192,"K":3,"wt":1})
    cfg = edict({"B":1,"T":5,"H":160,"W":160,"K":3,"wt":1})
    exps = fine_grid(cfg)
    exps = selected_grid(cfg)

    # -- run exps -
    reset = False
    df = cache_io.run_exps(exps,exec_method,clear=False,
                           name = ".cache_io/dev_compare_search/",
                           skip_loop=False,clear_fxn=None,
                           records_reload=False,to_records_fast=False,
                           use_wandb=False,enable_dispatch="slurm",
                           records_fn=".cache_io_pkl/dev_compare_search")
    df = df.drop_duplicates()

    # -- viz --
    df_f = df[df['F'] == 32].reset_index(drop=True)
    df_f = df_f[df_f['ps'] == 1].reset_index(drop=True)
    df_f = df_f[df_f['HD'] == 4].reset_index(drop=True)
    compare_memory(df_f)

    df_f = df[df['ws'] == 11].reset_index(drop=True)
    df_f = df_f[df_f['HD'] == 2].reset_index(drop=True)
    df_f = df_f[df_f['ps'] == 1].reset_index(drop=True)
    compare_times(df_f)


if __name__ == "__main__":
    main()
