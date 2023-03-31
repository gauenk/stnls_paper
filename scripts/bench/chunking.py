"""

Assess the impact of cropping.

"""
# -- misc --
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict

# -- cache --
import cache_io

# -- dev basics --
import dev_basics.exps as dev_exps

# -- network configs --
from icml23 import test_model
from icml23 import plots

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get list of exps; there are 6 total _files_ --
    exp_files = []
    category = ["spatial","temporal"]
    archs = ["colanet","n3net","lidia"]
    for arch in archs:
        for cat in category:
            exp_files.append("exps/%s_cropping_%s.cfg" % (cat,arch))

    # -- run/load experiments --
    records = cache_io.run_exps(exp_files,test_model.run,
                                name=".cache_io/chunking",
                                version="v1",skip_loop=True,
                                records_fn=".cache_io_pkl/chunking.pkl",
                                records_reload=False)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #          Plotting
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    #
    # -- only keep first processed video sequence --
    #

    filtered_df = []
    for _,_row in records.iterrows():
        row = _row.to_dict()
        row_f = {}
        for field,val in row.items():
            if isinstance(val,np.ndarray) and len(val) == 2:
                val = [val[0]]
            row_f[field] = val
        filtered_df.append(row_f)
    filtered_df = pd.DataFrame(filtered_df)
    records = filtered_df

    #
    # -- plot --
    #

    records['psnrs'] = records['psnrs'].apply(lambda x: x[0]).apply(np.mean)
    records['strred'] = records['strred'].apply(lambda x: x[0]).apply(np.mean)
    records['deno_mem_res'] = records['deno_mem_res'].apply(lambda x: np.mean(x))
    records['timer_deno'] = records['timer_deno'].apply(lambda x: np.mean(x))
    for arch,adf in records.groupby("arch_name"):
        print("plotting arch %s" % arch)
        plots.chunking.run(adf,arch)

if __name__ == "__main__":
    main()
