"""

Compare the impact of train/test using exact/refineimate methods


"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from dev_basics.trte import test,bench

# -- plotting --
import stnls_paper
# from stnls_paper import plots

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    refresh = False
    def clear_fxn(num,cfg):
        return False
    read_test = cache_io.read_test_config.run
    exps = read_test("exps/trte_nlnet/test.cfg",
                     ".cache_io_exps/trte_nlnet/test",reset=refresh,skip_dne=refresh)
    exps,uuids = cache_io.get_uuids(exps,".cache_io/trte_nlnet/test",
                                    no_config_check=refresh,reset=refresh)
    print("len(exps): ",len(exps))

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,
                                name=".cache_io/trte_nlnet/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_nlnet/test.pkl",
                                records_reload=False,to_records_fast=True)

    # -- get bench--
    # bench.print_summary(exps[0],(1,3,3,128,128))

    # -- view --
    print(len(results))
    if len(results) == 0: return
    afields = ['psnrs','ssims','strred']
    gfields = ["sigma","gradient_clip_val",'rate','nepochs']
    agg_fxn = lambda x: np.mean(np.stack(x))
    results = results.groupby(gfields).agg({k:agg_fxn for k in afields})
    for f in afields: results[f] = results[f].apply(np.mean)
    results = results.reset_index()[gfields + afields]
    print(len(results))
    print(results)

    # for sigma,sdf in results[afields + gfields].groupby("sigma"):
    #     print(sigma)
    #     print(sdf)
    # print(results[afields + ["vid_name","sigma"]])

if __name__ == "__main__":
    print("hi.")
    main()
