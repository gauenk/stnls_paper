"""

This script shows our backprop methods is x100 faster than deterministic cpu code.

"""

import os
import numpy as np

import cache_io
from stnls_paper.bench import compare_cpu
from stnls_paper import plots

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run cached records --
    exps = cache_io.get_exp_list("exps/bench/faster_than_cpu.cfg")
    print(len(exps))
    records = cache_io.run_exps(exps,compare_cpu.run_exp,
                                name=".cache_io/bench/",version="v1",
                                enable_dispatch="slurm",skip_loop=False,
                                records_fn=".cache_io_pkl/faster_than_cpu_agg.pkl",
                                records_reload=True,use_wandb=False,clear=False)

    # -- plot --
    print(len(records))
    plots.bench.faster_than_cpu.run(records)

if __name__ == "__main__":
    main()

