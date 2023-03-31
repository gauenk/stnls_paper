"""

Comparing the search

"""

# -- sys --
import os

# -- data mangle --
import pandas as pd

# -- exp --
from icml23 import plots
from icml23.agg import compare as compare_agg

# -- caching results --
import cache_io

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    results = cache_io.run_exps("exps/compare_agg.cfg",
                                compare_agg.run,
                                name=".cache_io/compare_agg",
                                version="v1",
                                skip_loop=True,
                                records_fn=".cache_io_pkl/compare_agg.pkl",
                                records_reload=True)

    # -- format times --
    results = pd.DataFrame(results)
    plots.compare_agg.run(results)

if __name__ == "__main__":
    main()
