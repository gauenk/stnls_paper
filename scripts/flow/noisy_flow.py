"""

Assess the impact of optical flow

"""


# -- sys --
import os

# -- caching results --
import cache_io

# -- data mangling --
import numpy as np

# -- network configs --
from icml23 import test_model
from icml23 import reports
from icml23 import plots

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- run experiments --
    records = cache_io.run_exps("exps/noisy_flow.cfg",
                                test_model.run,
                                name=".cache_io/noisy_flow",
                                version="v1",
                                skip_loop=True,
                                records_fn=".cache_io_pkl/noisy_flow.pkl",
                                records_reload=False)

    # -- compress metrics --
    ffields = {"k_s":[100]}
    gfields = ['sigma','wt','ws','flow_sigma']
    afields = ['psnrs','ssims','strred']
    for field,val in ffields.items():
        records = records[records[field].isin(val)]
    agg = lambda x: np.mean(np.stack(x))
    records = records.groupby(gfields).agg({k:agg for k in afields})
    records = records.reset_index()

    # -- plot --
    plots.noisy_flow.run(records)

if __name__ == "__main__":
    main()

