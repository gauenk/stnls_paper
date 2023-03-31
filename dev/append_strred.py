"""

Append strred by reading denoised output examples


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
from dev_basics.utils.metrics import compute_strred

# -- caching results --
import cache_io

# -- network configs --
from aaai23 import test_model
from aaai23 import reports

def find_missing(src,exps,skip_empty=False):
    uuids,configs,results = src.load_raw_configs(exps,skip_empty)
    if len(configs) == 0: warn_message(src)
    avail = []
    missing = []
    for cfg,uuid,result in zip(configs,uuids,results):
        info = {"vid_name":cfg.vid_name,"arch":cfg.arch_name,
                "dname":cfg.dname,"sigma":cfg.sigma,"wt":cfg.wt}
        miss_any = False
        valid_fns = not(result is None)
        if valid_fns:
            fns = result['deno_fns'][0]
            for fn in fns:
                fn = Path(fn)
                print(fn)
                if (not fn.exists()) or not(fn.suffix == ".png"):
                    miss_any = True
                    break
        else:
            print("result in none.")
            print(info)
        if miss_any or not(valid_fns):
            missing.append(info)
        else:
            avail.append(info)
    missing = pd.DataFrame(missing)
    missing.to_csv("missing.csv",index=False)
    avail = pd.DataFrame(avail)
    avail.to_csv("avail.csv",index=False)

def mod_copy(src,dest,exps,skip_empty=True):
    uuids,configs,results = src.load_raw_configs(exps,skip_empty)
    if len(configs) == 0: warn_message(src)
    mod_res = []
    N = len(configs)
    for cfg,result in tqdm.tqdm(zip(configs,results),total=N):
        fns = result['deno_fns'][0]
        deno = vid_io.read_files(fns)
        clean = get_clean(cfg)
        strred = compute_strred(clean,deno,255.)
        result['strred'] = [strred]
        mod_res.append(result)
    overwrite = True
    dest.save_raw(uuids,configs,mod_res,overwrite)

def get_clean(cfg):
    imax = 255.
    data,loaders = data_hub.sets.load(cfg)
    groups = data[cfg.dset].groups
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,
                                     cfg.frame_start,cfg.frame_end)
    clean = data[cfg.dset][indices[0]]['clean'].to(cfg.device)
    return clean

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- load cache --
    # cache_dir = ".cache_io/test_colanet_12_30_r2"
    # cache_dir = ".cache_io/test_lidia_noT"
    # cache_dir = ".cache_io/test_lidia_yesT"
    cache_dir = ".cache_io/test_nets"
    cache_name = "v1"
    cache0 = cache_io.ExpCache(cache_dir,cache_name)
    # cache_dir = ".cache_io/test_nets"
    # cache_name = "v1"
    # cache0 = cache_io.ExpCache(cache_dir,cache_name)

    # -- load dest cache --
    cache_dir = ".cache_io/test_nets_v2"
    cache_name = "v1"
    cache1 = cache_io.ExpCache(cache_dir,cache_name)

    # -- read exps --
    # exps = dev_exps.load("exps/test_colanet.cfg")
    exps = dev_exps.load("exps/test_n3net.cfg")
    # exps = dev_exps.load("exps/test_lidia.cfg")

    # -- run exps --
    # find_missing(cache0,exps)
    mod_copy(cache0,cache1,exps)

if __name__ == "__main__":
    main()
