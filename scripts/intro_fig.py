"""

Show modulation of GPU Memory and Runtime

"""


# -- misc --
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- cache --
import cache_io

# -- network configs --
from icml23 import test_model
from icml23 import plots

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- run experiments --
    records = cache_io.run_exps("exps/intro_fig.cfg",
                                test_model.run,
                                name=".cache_io/intro_fig_testing",
                                version="v1",
                                skip_loop=False,
                                records_fn=".cache_io_pkl/intro_fig_test.pkl",
                                records_reload=True)

    # -- process --
    aname = records['arch_name'].to_numpy()
    psnrs = records['psnrs'].apply(np.mean).to_numpy()
    dtimer = records['timer_deno'].apply(lambda x: x[0]).to_numpy()
    mem = records['deno_mem_res'].apply(lambda x: x[0][0]).to_numpy()
    bs = records['bs'].to_numpy()
    records['mem_res'] = records['deno_mem_res']

    labels = np.array(["" for _ in range(len(records))])
    labels = np.where(records['arch_name'] == "colanet","COLA-Net",labels)
    labels = np.where(records['arch_name'] == "n3net","N3Net",labels)
    labels = np.where(records['arch_name'] == "lidia","LIDIA",labels)
    records['label_name'] = labels
    plots.intro_fig.run(records)

if __name__ == "__main__":
    main()
