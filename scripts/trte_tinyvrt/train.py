"""

Compare the impact of train/test using exact/refineimate methods


"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from dev_basics.trte import train

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
    exps,uuids = cache_io.train_stages.run("exps/trte_tinyvrt/train.cfg",
                                           ".cache_io/trte_tinyvrt/train/")#,update=True)
                                           # load_complete=True,stage_select=0)
    # print(len(exps))
    # print(uuids[:5])
    # print(uuids[-5:])
    results = cache_io.run_exps(exps,train.run,uuids=uuids,
                                name=".cache_io/trte_tinyvrt/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_tinyvrt/train.pkl",
                                records_reload=True)


if __name__ == "__main__":
    main()
