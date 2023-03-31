"""

Comparing the search

"""


# -- sys --
import os

# -- experiment --
from icml23.search import compare
from icml23 import plots

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    clear_fxn = lambda x,y: False
    results = cache_io.run_exps("exps/compare_search.cfg",
                                compare.run,
                                name=".cache_io/compare_search",
                                version="v1",
                                skip_loop=False,
                                clear_fxn=clear_fxn,
                                records_fn=".cache_io_pkl/compare_search.pkl",
                                records_reload=True)

    # -- format times --
    print(results[['arch','mode','timer_fwd','timer_bwd']])
    print(results[['arch','mode','fwd_res','bwd_res']])
    plots.compare_search.run(results)

if __name__ == "__main__":
    main()
