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
                                           ".cache_io_exps/trte_f2f/train/")#,update=True)

    print(exps[0])
    print(len(exps))
    def clear_fxn(num,cfg): return True
    results = cache_io.run_exps(exps,train.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_f2f/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_f2f/train.pkl",
                                records_reload=False,use_wandb=True,
                                proj_name="neurips_f2f")
    # -- view --
    print(len(results))
    if len(results) == 0: return
    print(results.columns)
    # results = results[results['dset_tr'] == 'tr'].reset_index(drop=True)
    # results = results[results['dname'] == 'davis'].reset_index(drop=True)
    # results = results[results['limit_train_batches'] == 1.0].reset_index(drop=True)
    # print(results[['crit_name','ps','stride0','train_time']])

if __name__ == "__main__":
    main()

