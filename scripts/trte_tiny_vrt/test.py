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
    def clear_fxn(num,cfg):
        return False
    read_test = cache_io.read_test_config.run
    exps = read_test("exps/tiny_vrt/test.cfg",
                     cache_name=".cache_io_exps/tiny_vrt/test",
                     cache_reset=True)
    exps,uuids = cache_io.get_uuids(exps,".cache_io/tiny_vrt/test",
                                    no_config_check=False)

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,
                                name=".cache_io/tiny_vrt/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/tiny_vrt/test.pkl",
                                records_reload=True,to_records_fast=False)

    # print(len(results))
    if len(results) == 0: return
    mfields = ["vid_name","sigma",'gradient_clip_val','rate']
    afields = ['psnrs','ssims','strred']
    for f in afields: results[f] = results[f].apply(np.mean)
    results = results[afields + mfields]
    for sigma_rate,sdf in results.groupby(["sigma","rate","gradient_clip_val"]):
        print(sigma_rate)
        print(sdf)
        print({a:sdf[a].mean() for a in afields})
        # print(sdf[['psnrs'] + mfields])
    # print(results[afields + ["vid_name","sigma"]])

if __name__ == "__main__":
    main()
