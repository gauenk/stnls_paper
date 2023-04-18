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
    def clear_fxn(num,cfg): return True
    read_test = cache_io.read_test_config.run
    exps = read_test("exps/trte_colanet/test.cfg",
                     ".cache_io_exps/trte_colanet/test",reset=refresh,skip_dne=refresh)
    exps,uuids = cache_io.get_uuids(exps,".cache_io/trte_colanet/test",
                                    read=not(refresh),no_config_check=False)
    print(len(exps))

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,
                                name=".cache_io/trte_colanet/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_colanet/test.pkl",
                                records_reload=False,to_records_fast=False)

    # -- view --
    print(len(results))
    if len(results) == 0: return
    afields = ['psnrs','ssims','strred']
    gfields = ["sigma",'read_flows','wt','dname','k_s']
    agg_fxn = lambda x: np.mean(np.stack(x))
    for f in afields: results[f] = results[f].apply(np.mean)
    results = results.groupby(gfields).agg({k:agg_fxn for k in afields})
    results = results.reset_index()[gfields + afields]
    results = results.sort_values(["dname","sigma","read_flows"])
    results0 = results[results['k_s'] == 10].reset_index(drop=True)
    print(results0)
    results1 = results[results['k_s'] == 25].reset_index(drop=True)
    print(results1)
    # results = results.sort_values(["vid_name","dname","sigma","read_flows"])
    # results.to_csv("formatted_test_colanet.csv",index=False)
    # print(len(results))
    # print(results)

if __name__ == "__main__":
    main()
