"""

Compare the impact of train/test using exact/refineimate methods


"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from dev_basics.trte import test

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    read_test = cache_io.read_test_config.run
    exps = read_test("exps/trte_f2f/test.cfg",
                     cache_name=".cache_io_exps/trte_f2f/test")
    exps,uuids = cache_io.get_uuids(exps,".cache_io/trte_f2f/test",
                                    reset=True,no_config_check=True)

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,
                                name=".cache_io/trte_f2f/test",
                                version="v1",skip_loop=False,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_f2f/test.pkl",
                                records_reload=False,to_records_fast=True)

    # print(len(results))
    results = results.fillna("None")
    print(len(results))
    if len(results) == 0: return
    afields = ['psnrs','ssims','strred']
    gfields = ["search_input","crit_name","dist_crit","alpha",
               "ps","k","ws","wt","gradient_clip_val","dset_tr","sigma"]
    print(len(results[gfields].drop_duplicates()))
    # results = results[results['vid_name'] == "sunflower"].reset_index(drop=True)
    print(results)
    print(len(results))
    agg_fxn = lambda x: np.mean(np.stack(x))
    summary = results.groupby(gfields).agg({k:agg_fxn for k in afields})
    summary = summary.reset_index()[gfields + afields]
    print(summary)
    # print(summary['psnrs'].nlargest(10))
    # print(summary['psnrs'].nlargest(10).index)
    print(summary.iloc[summary['psnrs'].nlargest(30).index])

    # print(results[afields + ["vid_name","sigma"]])

if __name__ == "__main__":
    main()
