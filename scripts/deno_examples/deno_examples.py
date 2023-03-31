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


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    # cache_dir = ".cache_io"
    # cache_name = "test_nets"
    # cache = cache_io.ExpCache(cache_dir,cache_name)
    # exps = dev_exps.load("exps/test_colanet.cfg")
    # exps = [exp for exp in exps if exp.vid_name == "goat"]
    # exps = [exp for exp in exps if exp.sigma == 30]

    # -- get cache --
    cache_dir = ".cache_io/deno_examples/"
    # cache_name = "v1"
    cache_name = "v2"
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- run exps --
    # exps = dev_exps.load("exps/deno_examples_test_chunks.cfg")
    exps = dev_exps.load("exps/deno_examples_colanet.cfg")
    # exps += dev_exps.load("exps/deno_examples_lidia.cfg")
    # exps += dev_exps.load("exps/deno_examples_n3net.cfg")

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
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        exp.uuid = uuid
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            results = test_model.run(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    save_fn = ".cache_io_deno_examples.pkl"
    # save_fn = ".cache_io_stardeno.pkl"
    re_load = True
    records = cache.load_flat_records(exps,save_fn,re_load)

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
        deno_examples.run(vdf_ours,vdf_orig)


if __name__ == "__main__":
    main()
