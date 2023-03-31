
"""


"""


# -- misc --
import os,tqdm
import pprint,copy
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- dev basics --
import dev_basics.exps as dev_exps
from dev_basics.utils import vid_io

# -- caching results --
import cache_io

# -- network configs --
from icml23 import nl_approx_attn
from icml23 import plots

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- extract exps --
    exps = cache_io.get_exps("exps/nl_approx_attn_colanet.cfg")

    # -- partially initial function --
    exp_og = copy.deepcopy(exps[0])
    exp_og.model_type = "original"
    exp_og.k = -1
    fxn = lambda exp: nl_approx_attn.run(exp,exp_og)

    # -- get/run cached records --
    records = cache_io.run_exps(exps,fxn,
                                name=".cache_io/nl_approx_attn",
                                version="v1",
                                records_fn=".cache_io_pkl/nl_approx_attn.pkl",
                                records_reload=True,
                                skip_loop=True)
    # -- plot --
    plots.nl_approx_attn.run(records)


if __name__ == "__main__":
    main()
