"""

Non-Local Search approximates Attention

"""


# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- networks --
import colanet
import lidia
import n3net

# -- dev basics --
# from dev_basics.report import deno_report
from functools import partial
from dev_basics.aug_test import test_x8
from dev_basics import flow
from dev_basics import net_chunks
from dev_basics.utils.misc import optional,slice_flows,set_seed
from dev_basics.utils.misc import rslice,write_pickle,read_pickle
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.metrics import compute_psnrs,compute_ssims
from dev_basics.utils import vid_io

# -- config --
from functools import partial
from dev_basics.common import optional as _optional
from dev_basics.common import optional_fields,extract_config,extract_pairs
_fields = []
optional_full = partial(optional_fields,_fields)
extract_test_config = partial(extract_config,_fields)

def _extract_test_config(_cfg,optional):
    pairs = {"device":"cuda:0","seed":123,
             "frame_start":0,"frame_end":-1,
             "aug_test":False,"longest_space_chunk":False,
             "flow":False,"burn_in":False,"arch_name":None,
             "saved_dir":"./output","uuid":"uuid_def",
             "flow_sigma":-1,"internal_adapt_nsteps":0,
             "internal_adapt_nepochs":0,"nframes":0}
    return extract_pairs(pairs,_cfg,optional)

def run(cfg0,cfg1):

    # -- config --
    init = _optional(cfg0,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)
    tcfg0 = _extract_test_config(cfg0,optional)
    tcfg1 = _extract_test_config(cfg1,optional)
    if init: return

    # -- clear --
    th.cuda.empty_cache()
    th.cuda.synchronize()

    # -- set device --
    th.cuda.set_device(int(tcfg0.device.split(":")[1]))

    # -- set seed --
    set_seed(tcfg0.seed)

    # -- load model --
    model0 = load_model(cfg0)
    model1 = load_model(cfg1)

    # -- data --
    imax = 255.
    data,loaders = data_hub.sets.load(cfg0)
    groups = data[cfg0.dset].groups
    indices = data_hub.filter_subseq(data[cfg0.dset],cfg0.vid_name,
                                     tcfg0.frame_start,tcfg0.nframes)
    # -- compare --
    diff = 0
    errors = []

    # -- only indices --
    for index in indices:

        # -- unpack --
        sample = data[cfg0.dset][index]
        region = sample['region']
        noisy,clean = sample['noisy'],sample['clean']
        noisy,clean = noisy.to(tcfg0.device),clean.to(tcfg0.device)
        vid_frames = sample['fnums'].numpy()
        # print("[%d] noisy.shape: " % index,noisy.shape)

        # -- process --
        deno0 = model0.ca_forward(noisy/imax)
        deno1 = model1.ca_forward(noisy/imax)

        # -- compare --
        eps = 1e-4
        erel = th.abs(deno0 - deno1)/(th.abs(deno1)+eps)
        error = th.mean(erel).item()

        # -- append results --
        errors.append(error)

    # -- clear --
    th.cuda.empty_cache()
    th.cuda.synchronize()

    # -- format results --
    results = edict()
    results.error_m = np.mean(errors)
    results.error_s = np.std(errors)

    return results

# -- populate fields --
run({"__init":True},{"__init":True})

def run_sample(sample):
    # -- create timer --
    timer = ExpTimer()
    memer = GpuMemer()

    # -- clean memory --
    th.cuda.empty_cache()
    # print("index: ",index)

    # -- resample noise for flow --
    if tcfg.flow_sigma >= 0:
        noisy_f = th.normal(clean,tcfg.flow_sigma)
    else:
        noisy_f = noisy
    # print(th.std(noisy_f - clean).item())

    # -- optical flow --
    with TimeIt(timer,"flow"):
        flows = flow.orun(noisy_f,tcfg.flow)

    # -- internal adaptation --
    adapt_psnrs = [0.]
    run_adapt = tcfg.internal_adapt_nsteps > 0
    run_adapt = run_adapt and (tcfg.internal_adapt_nepochs > 0)
    with MemIt(memer,"adapt"):
        with TimeIt(timer,"adapt"):
            if run_adapt:
                noisy_a = noisy[:5]
                clean_a = clean[:5]
                flows_a = flow.slice_at(flows,slice(0,5),1)
                region_gt = get_region_gt(noisy_a.shape)
                adapt_psnrs = model.run_internal_adapt(
                    noisy_a,cfg.sigma,flows=flows,
                    clean_gt = clean_a,region_gt = region_gt)

    # -- denoise --
    if tcfg.aug_test:
        aug_fxn = partial(test_x8,model)#,use_refine=cfg.aug_refine_inds)
    else:
        aug_fxn = model

    # -- set chunks --
    chunk_cfg = net_chunks.extract_chunks_config(cfg)
    if tcfg.longest_space_chunk:
        set_longest_spatial_chunk(chunk_cfg,noisy.shape)
    fwd_fxn = net_chunks.chunk(chunk_cfg,aug_fxn)

    # -- run once for setup gpu --
    if tcfg.burn_in:
        with th.no_grad():
            noisy_a = noisy[[0],...,:128,:128].contiguous()
            flows_a = flow.orun(noisy_a,False)
            fwd_fxn(noisy_a/imax,flows_a)
        if hasattr(model,'reset_times'):
            model.reset_times()

    # -- benchmark it! --
    with MemIt(memer,"deno"):
        with TimeIt(timer,"deno"):
            with th.no_grad():
                deno = fwd_fxn(noisy/imax,flows)
            deno = deno.clamp(0.,1.)*imax
    mtimes = model.times

    # -- unpack if exists --
    if hasattr(model,'mem_res'):
        if model.mem_res != -1:
            memer["deno"] = (model.mem_res,model.mem_alloc)

    # -- save example --
    out_dir = Path(tcfg.saved_dir) / str(tcfg.uuid)
    deno_fns = vid_io.save_burst(deno,out_dir,"deno")

    # -- psnr --
    noisy_psnrs = compute_psnrs(noisy,clean,div=imax)
    psnrs = compute_psnrs(deno,clean,div=imax)
    ssims = compute_ssims(deno,clean,div=imax)


def load_model(cfg):
    if cfg.arch_name == "colanet":
        return colanet.load_model(cfg)
    elif cfg.arch_name == "lidia":
        return lidia.load_model(cfg)
    elif cfg.arch_name == "n3net":
        return n3net.load_model(cfg)
    else:
        raise ValueError(f"Uknown arch_name [{arch_name}]")


# -- used for adaptation --
def get_region_gt(vshape):

    t,c,h,w = vshape
    hsize = min(h//4,128)
    wsize = min(w//4,128)
    tsize = min(t//4,5)

    t_start = max(t//2 - tsize//2,0)
    t_end = min(t_start + tsize,t)
    if t == 3: t_start = 0

    h_start = max(h//2 - hsize//2,0)
    h_end = min(h_start + hsize,h)

    w_start = max(w//2 - wsize//2,0)
    w_end = min(w_start + wsize,w)

    region_gt = [t_start,t_end,h_start,w_start,h_end,w_end]
    return region_gt

