"""

Compare the impact of train/test using exact/refineimate methods


"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from dev_basics.trte import test

# -- plotting --
# import stnls_paper
# from stnls_paper import plots

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    def clear_fxn(num,cfg):
        return False
    read_test = cache_io.read_test_config.run
    exps = read_test("exps/trte_tinyvrt/test.cfg",
                     cache_name=".cache_io_exps/trte_tinyvrt/test")
    exps,uuids = cache_io.get_uuids(exps,".cache_io/trte_tinyvrt/test")
    # print(len(exps))

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,
                                name=".cache_io/trte_tinyvrt/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_tinyvrt/test.pkl",
                                records_reload=False,to_records_fast=True)

    print(len(results))
    if len(results) == 0: return
    results = results[results["spynet_path"] != ""].reset_index(drop=True)
    afields = ['psnrs','ssims','strred']
    gfields = ["sigma","gradient_clip_val",'rate']
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
    main()
