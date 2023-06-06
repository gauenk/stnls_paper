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

    # -- get experiments --
    def clear_fxn(num,cfg): return False
    exps,uuids = cache_io.train_stages.run("exps/trte_nlnet/train.cfg",
                                           ".cache_io_exps/trte_nlnet/train/",
                                           update=True)
    uuids = ["8b8aa3a4-e197-4857-9dbc-3bf11fa462d3",
             "5bac0f65-1828-40df-bcd2-1c8ab9169362",
             "c6bcb0b5-c5ba-4a7c-bf00-1d2e45fa2b57"]

    # -- run exps --
    results = cache_io.run_exps(exps,train.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_nlnet/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_nlnet/train.pkl",
                                records_reload=False,use_wandb=True,
                                proj_name="nlnet_train")



if __name__ == "__main__":
    main()
