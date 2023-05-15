"""

This graphic shows the errors incurred by the race condition

"""

import os
import numpy as np

import cache_io
from stnls_paper import race_cond
# from stnls_paper import plots

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run cached records --
    exps = cache_io.get_exp_list("exps/race_cond/race_cond.cfg")
    print(len(exps))
    records = cache_io.run_exps(exps,race_cond.run_exp,
                                name=".cache_io/race_cond",
                                version="v1",skip_loop=False,
                                records_fn=".cache_io_pkl/race_cond_agg.pkl",
                                records_reload=True,use_wandb=False,
                                clear=True)

    # -- viz --
    print(records[['errors_m_0','errors_m_1','dtime','exact_time','nchnls']])
    # print(records)
    print("done.")
    return
    # -- plot --
    plots.race_cond.run(records)

    # -- plot time vs error --
    fields = {"query_pt":2,"neigh_pt":2}
    records_f = records
    for field,val in fields.items():
        records_f = records_f[records_f[field] == val]
    plots.race_cond_v2.run(records_f)

if __name__ == "__main__":
    main()

