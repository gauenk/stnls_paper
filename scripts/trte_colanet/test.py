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
                                    read=True,no_config_check=refresh,force_read=True)
    print(len(exps))
    # print(exps[0])

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
    results = results[results['rbwd'] != False].reset_index(drop=True)
    results = results[results['sigma'] != 15].reset_index(drop=True)
    results = results.rename(columns={"gradient_clip_val":"gcv"})
    results = results[results['gcv'] != 0].reset_index(drop=True)
    afields = ['psnrs','ssims','strred']
    # gfields = ["sigma","gradient_clip_val",'read_flows','wt','rbwd','dname']
    gfields = ["sigma",'dname','wt','read_flows','rbwd','gcv','k_s']
    agg_fxn = lambda x: np.mean(np.stack(x))
    for f in afields: results[f] = results[f].apply(np.mean)
    results = results.groupby(gfields).agg({k:agg_fxn for k in afields})
    results = results.reset_index()[gfields + afields]
    results = results.sort_values(["dname","sigma","read_flows"])
    print(len(results))
    # print(results['k_s'])
    # results0 = results[results['k_s'] == 10].reset_index(drop=True)

    sort_fields = ['sigma','dname','wt','read_flows','rbwd','gcv']
    dnames = results['dname'].unique()
    for dname in dnames:
        results_d = results[results['dname'] == dname].reset_index(drop=True)
        results0 = results_d
        print(results0.sort_values(sort_fields))
        # results1 = results_d[results_d['k_s'] == 30].reset_index(drop=True)
        # print(results1.sort_values(sort_fields))
    # results = results.sort_values(["vid_name","dname","sigma","read_flows"])
    # results.to_csv("formatted_test_colanet.csv",index=False)
    # print(len(results))
    # print(results)

    # -- format results --
    df_set8 = results[results['dname'] == "set8"]
    report = reports.deno_table_v3.run_latex(df_set8)
    print(report)

    df_davis = results[results['dname'] == "davis"]
    report = reports.deno_table_v3.run_latex(df_davis)
    print(report)


if __name__ == "__main__":
    main()
