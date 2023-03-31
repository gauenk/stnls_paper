"""

Run models to get deno examples

"""

# -- misc --
import os,math,tqdm
import pprint,random,copy
pp = pprint.PrettyPrinter(indent=4)
from functools import partial

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- dev basics --
import dev_basics.exps as dev_exps
from dev_basics.reports import deno_report
from dev_basics.utils import vid_io

# -- caching results --
import cache_io

# -- network configs --
from icml23 import test_model
from icml23 import reports
from icml23 import deno_examples

def load_res(root,name,in_exps):

    # -- init --
    cache = cache_io.ExpCache(root,name)

    # -- filer --
    _exps = []
    for e in in_exps:
        if "spatial_chunk_overlap" in e:
            if e.spatial_chunk_overlap > 0: continue
            if e['spatial_chunk_overlap'] == 0 and name == "v2":
                del e['spatial_chunk_overlap']
        _exps.append(e)

    # -- collect --
    rec = cache.to_records(_exps)
    return rec

def load_all_res(cfg):
    in_exps = dev_exps.load(cfg.exps_fn)
    in_exps = filter_exps(cfg,in_exps)
    rec = []
    for root,name in zip(cfg.cache_root,cfg.cache_name):
        rec_r = load_res(root,name,in_exps)
        rec.append(rec_r)
    rec = pd.concat(rec)
    return rec

def filter_exps(cfg,in_exps):
    fields = ["vid_name","sigma"]
    for field in fields:
        if len(in_exps) == 0: break
        in_exps = [e for e in in_exps if e[field] == cfg[field]]
    if len(in_exps) == 0:
        msg = "Filtered exps are zero so no deno examples saved."
        print(msg)
        exit(0)
    return in_exps

def save_deno_example(cfg):

    # -- load all results --
    records = load_all_res(cfg)

    # -- only have two --
    assert len(records) == 2,f"Should only be two [{len(records)}]"

    # -- save --
    method = np.where(records['wt'] == 3,"Ours","Original")
    records['method'] = method
    records = records.sort_values(["vid_name","method"])
    records = records.reset_index(drop=True)
    for vid_name,vdf in records.groupby("vid_name"):
        vdf = vdf.sort_values("method")
        vdf.reset_index(inplace=True,drop=True)
        vdf_orig = edict(vdf.loc[0].to_dict())
        vdf_ours = edict(vdf.loc[1].to_dict())
        deno_examples.run(vdf_ours,vdf_orig,cfg.nframes,cfg.frame_start)

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    # cache_dir = ".cache_io/deno_examples/"
    # cache_name = "v3"
    # cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- run exps --
    exps = dev_exps.load("exps/deno_examples_lidia.cfg")
    print(exps)

    # -- run --
    nexps = len(exps)
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        # uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        # exp.uuid = uuid
        # results = cache.load_exp(exp) # possibly load result
        # if results is None: # check if no result
        results = save_deno_example(exp)
        # cache.save_exp(uuid,exp,results) # save to cache

if __name__ == "__main__":
    main()
