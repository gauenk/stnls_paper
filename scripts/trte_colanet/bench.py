"""

Benchmark different nets

"""
# -- sys --
import os

# -- caching results --
import cache_io

# -- network configs --
from stnls_paper import reports

# -- bench --
from dev_basics.trte import bench


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- records --
    def clear_fxn(num,cfg): return False
    exps,uuids = cache_io.train_stages.run("exps/trte_colanet/bench.cfg",
                                           ".cache_io/trte_colanet/bench/",
                                           update=True)

    # print(uuids)
    print("Num Exps: ",len(exps))
    results = cache_io.run_exps(exps,bench.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_colanet/bench",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_colanet/bench.pkl",
                                records_reload=True,
                                use_wandb=False,proj_name="tr_colanet")
    # -- view --
    if len(results) == 0: return
    print(results.head())
    fields = ['search_v0','wt','timer_fwd_nograd','res_fwd_nograd','alloc_fwd_nograd']
    print(results[fields])
if __name__ == "__main__":
    main()
