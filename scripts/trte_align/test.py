"""

Compare the impact of train/test using exact/refineimate methods


"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from superpixel_paper.deno_trte import test
# from dev_basics.trte import test,bench

# -- plotting --
import stnls_paper
# from stnls_paper import plots

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    refresh = False
    def clear_fxn(num,cfg): return False
    read_test = cache_io.read_test_config.run
    exps = read_test("exps/trte_align/test.cfg",
                     ".cache_io_exps/trte_align/test",
                     reset=refresh,skip_dne=refresh)
    exps,uuids = cache_io.get_uuids(exps,".cache_io/trte_align/test",
                                    no_config_check=False,update=refresh)
    print("len(exps): ",len(exps))
    # print([e.wt for e in exps if e.vid_name == "sunflower"])


    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,#preset_uuids=True,
                                name=".cache_io/trte_align/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_align/test.pkl",
                                records_reload=True,to_records_fast=False,
                                use_wandb=True,proj_name="test_align")

    # -- get bench--
    # bench.print_summary(exps[304],(1,3,3,128,128))

    # -- view --
    print(len(results))
    if len(results) == 0: return
    # results = results[results['input_proj_depth'] == 1].reset_index(drop=True)
    # results = results[results['read_flows'] == True].reset_index(drop=True)
    # results = results[results['embed_dim'] == 12].reset_index(drop=True)
    # results = results.rename(columns={"gradient_clip_val":"gcv"})
    afields = ['psnrs','ssims','strred']
    # gfields = ["sigma",'label0','dname','embed_dim','ws','pretrained_path']
    gfields = ["sigma",'vid_name']#'pretrained_path']
    # gfields = ["gcv",'ws','nheads','pretrained_path','sigma','dname',
    #            'read_flows','label0']
    # gfields = ["gcv",'ws','nheads','pretrained_path','sigma','dname',
    #            'read_flows','label0']
    # gfields0 = ["gcv",'ws','dname','label0']
    gfields0 = [gfields[i] for i in range(len(gfields)-1)]
    agg_fxn = lambda x: np.mean(x)
    for f in afields: results[f] = results[f].apply(np.mean)
    results = results.groupby(gfields).agg({k:agg_fxn for k in afields})
    results = results.reset_index()[gfields + afields]
    print(len(results))
    print(results)
    print(results['psnrs'].mean(),results['ssims'].mean())
    # results0 = results[results['sigma'] == 30]
    # results0 = results0[results0['label0'] == "(300,30)"]
    # print(results0.sort_values(["gcv","dname","embed_dim"])[gfields0+afields])

    # results0 = results[results['sigma'] == 30]
    # results0 = results0[results0['label0'] == "(300,50)"]
    # print(results0.sort_values(["gcv","dname","embed_dim"])[gfields0+afields])


    # results0 = results[results['sigma'] == 50]
    # results0 = results0[results0['label0'] == "(300,30)"]
    # print(results0.sort_values(["gcv","dname","embed_dim"])[gfields0+afields])

    # results0 = results[results['sigma'] == 50]
    # results0 = results0[results0['label0'] == "(300,50)"]
    # print(results0.sort_values(["gcv","dname","embed_dim"])[gfields0+afields])

    # results0 = results[results['sigma'].isin([30,50])]
    # results0 = results0[results0['label0'].isin(["(300,30)","(300,50)"])]
    # results0 = results0[results0['gcv'] == 0.5]
    # results0 = results0[results0['ws'] == 9]
    # print(results0.sort_values(["gcv","dname","embed_dim"])[gfields0+afields])


if __name__ == "__main__":
    main()
