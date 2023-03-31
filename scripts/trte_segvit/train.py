"""

Fine-tune the segmentation model from detectronv2

"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from dev_basics.trte import train

# -- plotting --
from apsearch import plots

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
    exps,uuids = cache_io.train_stages.run("exps/seg_vit/train.cfg",
                                           ".cache_io/seg_vit/train/",
                                           load_complete=True,stage_select=0)
    results = cache_io.run_exps(exps,train.run,uuids=uuids,
                                name=".cache_io/seg_vit/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/seg_vit/train.pkl",
                                records_reload=True)


if __name__ == "__main__":
    main()