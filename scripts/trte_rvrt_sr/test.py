"""

Compare the impact of train/test using exact/refineimate methods


"""


# -- sys --
import os
import numpy as np
import pandas as pd
from easydict import EasyDict as edict

# -- testing --
from dev_basics.trte import test,bench

# -- plotting --
# import stnls_paper
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
    exps = read_test("exps/trte_rvrt_sr/test.cfg",
                     ".cache_io_exps/trte_rvrt_sr/test",reset=refresh,skip_dne=refresh)
    exps,uuids = cache_io.get_uuids(exps,".cache_io/trte_rvrt_sr/test",
                                    reset=refresh,no_config_check=refresh)
    print(len(exps))

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,
                                name=".cache_io/trte_rvrt_sr/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_rvrt_sr/test.pkl",
                                records_reload=False,to_records_fast=True,
                                proj_name="test_rvrt_sr")

    # -- get bench--
    # bench.print_summary(exps[0],(1,4,4,256,256))

    # # -- view --
    # print(results['nepochs'].unique())
    # print(len(results))
    # if len(results) == 0: return
    # # print(results['pretrained_path'].unique())
    # results = results.rename(columns={"gradient_clip_val":"gcv"})
    # results = results[results["spynet_path"] != ""].reset_index(drop=True)
    # results = results[results["rate"] == -1].reset_index(drop=True)
    # afields = ['psnrs','ssims','strred']
    # gfields = ["sigma","gcv",'nepochs','label0','dname','tr_uuid','pretrained_path']
    # gfields0 = [gfields[i] for i in range(len(gfields)-2)]
    # agg_fxn = lambda x: np.mean(np.stack(x))
    # for f in afields: results[f] = results[f].apply(np.mean)
    # results = results.groupby(gfields).agg({k:agg_fxn for k in afields})
    # results = results.reset_index()[gfields + afields]
    # print(len(results))

    # results0 = results[results['nepochs'] == 200]
    # results0 = results0[results0['sigma'] == 30]
    # print(results0[gfields0+afields])
    # results0 = results[results['nepochs'] == 200]
    # results0 = results0[results0['sigma'] == 50]
    # print(results0[gfields0+afields])

    # results0 = results[results['nepochs'] == 300]
    # results0 = results0[results0['sigma'] == 30]
    # print(results0[gfields0+afields])
    # results0 = results[results['nepochs'] == 300]
    # results0 = results0[results0['sigma'] == 50]
    # print(results0[gfields0+afields])

    # print(results['tr_uuid'])

    # for sigma,sdf in results[afields + gfields].groupby("sigma"):
    #     print(sigma)
    #     print(sdf)
    # print(results[afields + ["vid_name","sigma"]])

if __name__ == "__main__":
    main()
