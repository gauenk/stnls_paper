"""

   I had a data_hub.common issue so the filter ran two subseq instead of one.
   This script picks the first index.
   The qualitative comparisons will be misaligned with half of colanet.

"""


# -- misc --
import os,tqdm
import pprint
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- dev basics --
import dev_basics.exps as dev_exps
from dev_basics.utils import vid_io

# -- caching results --
import cache_io

# -- network configs --
from aaai23 import test_model
from aaai23 import plots


def copy_modded(src,dest,configs,skip_empty=True,overwrite=False):
    uuids,configs,results = src.load_raw_configs(configs,skip_empty)
    if len(configs) == 0: warn_message(src)
    for result in results:
        for field in result.keys():
            if len(result[field]) == 2:
                result[field] = result[field][[0]]
    dest.save_raw(uuids,configs,results,overwrite)

def main():

    # -- get cache --
    cache_dir = ".cache_io/scaling_search"
    cache_name = "v1"
    cache0 = cache_io.ExpCache(cache_dir,cache_name)

    cache_dir = ".cache_io/scaling_search_v3"
    cache_name = "v1"
    cache1 = cache_io.ExpCache(cache_dir,cache_name)

    # -- only pick first one --
    exps = dev_exps.load("exps/scaling_search_colanet.cfg")
    exps += dev_exps.load("exps/scaling_search_n3net.cfg")
    exps += dev_exps.load("exps/scaling_search_lidia.cfg")
    copy_modded(cache0,cache1,exps)

    # -- check --
    records = cache1.to_records(exps) 
    print(len(exps),len(records))

if __name__ == "__main__":
    main()
