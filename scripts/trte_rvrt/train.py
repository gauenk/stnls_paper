"""

Compare the impact of train/test using exact/refineimate methods


"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from dev_basics.trte import train,bench

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    def clear_fxn(num,cfg): return False
    exps,uuids = cache_io.train_stages.run("exps/trte_rvrt/train.cfg",
                                           ".cache_io/trte_rvrt/train/",
                                           update=True)
    print(len(exps))

    # -- get bench--
    # bench.print_summary(exps[0],(1,3,3,128,128))
    results = cache_io.run_exps(exps,train.run,uuids=uuids,
                                name=".cache_io/trte_rvrt/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_rvrt/train.pkl",
                                records_reload=True,proj_name="rvrt_train")


if __name__ == "__main__":
    main()
