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
                     ".cache_io_exps/trte_colanet/test",
                     reset=refresh,skip_dne=refresh)
    exps,uuids = cache_io.get_uuids(exps,".cache_io/trte_colanet/test",
                                    read=False,no_config_check=False)
    print(len(exps))
    # print(exps[0])

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,
                                name=".cache_io/trte_colanet/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_colanet/test.pkl",
<<<<<<< HEAD
                                records_reload=False,to_records_fast=False)
=======
                                records_reload=False,to_records_fast=True)
>>>>>>> efedfda0fe8089dc1eee98034e7ca612455d3908

    # -- view --
    print(len(results))
    if len(results) == 0: return
    results = results[results['sigma'] != 15].reset_index(drop=True)
    results = results.rename(columns={"gradient_clip_val":"gcv"})
    afields = ['psnrs','ssims','strred']
<<<<<<< HEAD
    gfields = ["sigma","gradient_clip_val",'read_flows','wt','rbwd','dname']
=======
    gfields = ["sigma",'dname','wt','read_flows','rbwd','gcv','k_s']
>>>>>>> efedfda0fe8089dc1eee98034e7ca612455d3908
    agg_fxn = lambda x: np.mean(np.stack(x))
    for f in afields: results[f] = results[f].apply(np.mean)
    results = results.groupby(gfields).agg({k:agg_fxn for k in afields})
    results = results.reset_index()[gfields + afields]
    results = results.sort_values(["dname","sigma","read_flows"])
    results0 = results[results['k_s'] == 10].reset_index(drop=True)
    sort_fields = ['sigma','dname','wt','read_flows','rbwd','gcv']
    print(results0.sort_values(sort_fields))
    results1 = results[results['k_s'] == 25].reset_index(drop=True)
    print(results1.sort_values(sort_fields))
    # results = results.sort_values(["vid_name","dname","sigma","read_flows"])
    # results.to_csv("formatted_test_colanet.csv",index=False)
    # print(len(results))
    # print(results)

if __name__ == "__main__":
    main()
