"""

This graphic shows the errors incurred by the race condition

"""

import os
import numpy as np

import cache_io
from stnls_paper.bench import compare_st_search as compare
from stnls_paper import plots

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run cached records --
    exps = cache_io.get_exp_list("exps/bench/compare_st_search.cfg")
    print(len(exps))
    records = cache_io.run_exps(exps,compare.run_exp,
                                name=".cache_io/bench/",version="v1",
                                enable_dispatch="slurm",skip_loop=False,
                                records_fn=".cache_io_pkl/compare_st_search.pkl",
                                records_reload=False,use_wandb=False,clear=False)

    # -- plot --
    print(len(records))
    df = records[records['use_simp'] == True].reset_index(drop=True)
    plots.bench.backward_errors.run(df)
    plots.bench.faster_than_cpu.run(df)
    # df = records[records['use_simp'] == True].reset_index(drop=True)
    plots.bench.saving_mem.run(records)

if __name__ == "__main__":
    main()

