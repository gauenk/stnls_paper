"""

Test the finetuned colanet models

"""

# -- sys --
import os

# -- caching results --
import cache_io

# -- data mangling --
import numpy as np

# -- network configs --
from stnls_paper import reports

# -- bench --
from dev_basics.trte import test


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    refresh = False
    def clear_fxn(num,cfg): return False
    read_test = cache_io.read_test_config.run
    exps = read_test("exps/trte_colanet/test.cfg",
                     ".cache_io_exps/trte_colanet/test",reset=refresh,skip_dne=refresh)
    exps,uuids = cache_io.get_uuids(exps,".cache_io/trte_colanet/test",
                                    reset=refresh,no_config_check=refresh)
    print(len(exps))

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,
                                name=".cache_io/trte_colanet/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_colanet/test.pkl",
                                records_reload=True,to_records_fast=True)

    # -- view --
    print(len(results))
    if len(results) == 0: return
    afields = ['psnrs','ssims','strred']
    gfields = ["sigma","gradient_clip_val",'read_flow','wt','rbwd']
    agg_fxn = lambda x: np.mean(np.stack(x))
    results = results.groupby(gfields).agg({k:agg_fxn for k in afields})
    for f in afields: results[f] = results[f].apply(np.mean)
    results = results.reset_index()[gfields + afields]
    print(len(results))
    print(results)

if __name__ == "__main__":
    main()
