"""

Measure computational resouces consumed when scaling resolution.

"""

# -- misc --
import os

# -- caching results --
import cache_io

# -- network configs --
from icml23 import test_model
from icml23 import plots

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run cached records --
    exps= ["exps/resolution_scaling/resolution_scaling_colanet.cfg",
           "exps/resolution_scaling/resolution_scaling_n3net.cfg",
           "exps/resolution_scaling/resolution_scaling_lidia.cfg"]
    records_fn = ".cache_io_pkl/resolution_scaling_agg.pkl"
    records = cache_io.run_exps(exps,test_model.run,
                                name=".cache_io/resolution_scaling",
                                version="v1",skip_loop=True,
                                records_fn=records_fn,
                                records_reload=True)

    # -- plot --
    plots.resolution_scaling.run(records)

if __name__ == "__main__":
    main()
