"""

Check runtimes

"""

# -- basic --
import torch as th
import numpy as np

# -- natten --
from natten import NeighborhoodAttention2D
from natten.functional import natten2dav, natten2dqkrpb

# -- bench --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from dev_basics import flow

# -- stnls --
import stnls


def run_natten(q,k,v,ksize=3,rpb=None,dilation=1):
    attn = natten2dqkrpb(q, k, None, ksize, dilation)
    attn = attn.softmax(dim=-1)
    topk = th.topk(attn,3,-1,largest=True)
    dists,inds = topk.values,topk.indices
    # print(inds)
    # print(th.topk(attn,1,3))
    # x = natten2dav(attn, v, dilation)
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

# def get_nat_data(B,HD,T,F,H,W,St):
#     q = th.randn((B*T*St,H,W,1,HD,F),device="cuda").permute(3, 0, 4, 1, 2, 5)[0]
#     k = th.randn((B*T*St,H,W,1,HD,F),device="cuda").permute(3, 0, 4, 1, 2, 5)[0]
#     v = th.randn((B*T*St,H,W,1,HD,F),device="cuda").permute(3, 0, 4, 1, 2, 5)[0]
#     return q,k,v

# def get_stnls_data(B,HD,T,F,H,W):
#     q = th.randn((B,HD,T,F,H,W),device="cuda")
#     k = th.randn((B,HD,T,F,H,W),device="cuda")
#     v = th.randn((B,HD,T,F,H,W),device="cuda")
#     return q,k,v

def main():

    # -- config --
    rpb = None
    B = 1
    T = 5
    F = 32
    # H = 960//2
    # W = 480//2
    H = 160
    W = 160

    wt = 1
    ksize = 3
    ws = ksize
    K = 3
    ps = 1
    stride0 = 2
    HD = 2

    # -- init --
    timer = ExpTimer()
    memer = GpuMemer()

    # -- n3net --
    ws = 9
    q,k,v = get_stnls_data(B,HD,T,F,H,W)
    zflow = th.zeros((B,T,2,H,W),device="cuda")
    with TimeIt(timer,"n3net"):
        with MemIt(memer,"n3net"):
            run_n3net(q,k,v,zflow,ws,wt,ps,K,stride0,HD)

    # -- natten --
    St = 2*wt+1
    q,k,v = get_nat_data(B,HD,T,F,H,W,St)
    print("nat: ",q.shape)
    with TimeIt(timer,"nat"):
        with MemIt(memer,"nat"):
            run_natten(q,k,v,ksize,rpb)

    # -- stnls --
    ws = 9
    q,k,v = get_stnls_data(B,HD,T,F,H,W)
    zflow = th.zeros((B,T,2,H,W),device="cuda")
    print("stnls: ",q.shape)
    with TimeIt(timer,"stnls"):
        with MemIt(memer,"stnls"):
            run_stnls(q,k,v,zflow,ws,wt,ps,K,stride0,HD)

    # # -- stnls --
    # ws = 33
    # q,k,v = get_stnls_data(B,HD,T,F,H,W)
    # zflow = th.zeros((B,T,2,H,W),device="cuda")
    # with TimeIt(timer,"stnls_big"):
    #     with MemIt(memer,"stnls_big"):
    #         run_stnls(q,k,v,zflow,ws,wt,ps,K,stride0,HD)

    # -- stnls --
    ws = 1
    q,k,v = get_stnls_data(B,HD,T,F,H,W)
    zflow = th.zeros((B,T,2,H,W),device="cuda")
    with TimeIt(timer,"stnls_small"):
        with MemIt(memer,"stnls_small"):
            run_stnls(q,k,v,zflow,ws,wt,ps,K,stride0,HD)


    # # -- flows --
    # ws = 9
    # ksize = 1
    # q,k,v = get_stnls_data(B,HD,T,F,H,W)
    # zflow = th.zeros((B,T,2,H,W),device="cuda")
    # with TimeIt(timer,"flow"):
    #     with MemIt(memer,"flow"):
    #         flows = flow.orun(q[:,0],True,ftype="cv2")

    # # -- nn --
    # ws = 9
    # ksize = 1
    # q,k,v = get_stnls_data(B,HD,T,F,H,W)
    # zflow = th.zeros((B,T,2,H,W),device="cuda")
    # with TimeIt(timer,"nn"):
    #     with MemIt(memer,"nn"):
    #         run_conv2d(q,k,v,ksize)#zflow,ws,wt,ps,K,stride0,HD)

    # -- viz --
    print(timer)
    print(memer)
    print(timer['stnls'] / timer['nat'])
    print(timer['n3net'] / timer['stnls'])


if __name__ == "__main__":
    main()
