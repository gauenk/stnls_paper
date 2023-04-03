
# -- exp --
import torch as th
from .api import init_search
from easydict import EasyDict as edict
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from dev_basics.utils.misc import set_seed

def run(cfg):

    # -- init seed --
    set_seed(cfg.seed)

    # -- init search --
    search = init_search(cfg,cfg.arch,cfg.mode)

    # -- init --
    timer = ExpTimer()
    memer = GpuMemer()

    # -- alloc --
    vid = th.rand((1,cfg.T,cfg.chnls,cfg.H,cfg.W),device="cuda:0")
    vid.requires_grad_(True)

    # -- first launch is always slower --
    _,_ = search(vid)
    th.cuda.synchronize()

    # -- forward --
    name = "fwd"
    with TimeIt(timer,name):
        with MemIt(memer,name):
            dists,inds = search(vid)

    # -- backward --
    name = "bwd"
    dists_grad = th.randn_like(dists)
    with TimeIt(timer,name):
        with MemIt(memer,name):
            th.autograd.backward(dists,dists_grad)

    # -- format times --
    results = edict()
    for key,val in timer.items():
        results[key] = val
    for key,(res,alloc) in memer.items():
        for f,mem in zip(["res","alloc"],[res,alloc]):
            res_key = "%s_%s" % (key,f)
            results[res_key] = mem
    return results

