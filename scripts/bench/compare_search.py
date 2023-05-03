"""

Comparing the search

"""


# -- sys --
import os

# -- experiment --
# from stnls_paper.search import compare
from stnls_paper.search import compare_module
from stnls_paper import plots

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    clear_fxn = lambda x,y: False
    results = cache_io.run_exps("exps/bench/compare_search.cfg",
                                compare_module.run,
                                name=".cache_io/bench/compare_search",
                                version="v1",skip_loop=False,
                                clear_fxn=clear_fxn,
                                records_fn=".cache_io_pkl/bench/compare_search.pkl",
                                records_reload=True)

    # -- format times --
    print(results[['arch','mode','timer_fwd','timer_bwd']])
    print(results[['arch','mode','fwd_res','bwd_res']])
    plots.compare_search.run(results)

if __name__ == "__main__":
    main()
