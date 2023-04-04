"""

Unsupervised Training with Frame2Frame

Compare the impact of train/test using flow/nls methods

"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- clearing --
import shutil
from pathlib import Path

# -- testing --
from dev_basics.trte import train
# import frame2frame

# -- caching results --
import cache_io

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    exps,uuids = cache_io.train_stages.run("exps/trte_f2f/train.cfg",
                                           ".cache_io_exps/trte_f2f/train/")
    def clear_fxn(num,cfg): return False
    results = cache_io.run_exps(exps,train.run,uuids=uuids,
                                name=".cache_io/trte_f2f/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_f2f/train.pkl",
                                records_reload=False)


if __name__ == "__main__":
    main()

