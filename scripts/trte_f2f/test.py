"""

Compare the impact of train/test using exact/refineimate methods


"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
# from dev_basics.trte import test
from frame2frame import test

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
    exps = read_test("exps/trte_f2f/test.cfg",
                     ".cache_io_exps/trte_f2f/test",reset=refresh,skip_dne=refresh)
    exps,uuids = cache_io.get_uuids(exps,".cache_io/trte_f2f/test",
                                    read=not(refresh),no_config_check=False)
    print(uuids)
    print("Run Exps: ",len(exps))

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_f2f/test",
                                version="v1",skip_loop=False,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_f2f/test.pkl",
                                records_reload=False,to_records_fast=True,
                                use_wandb=True,proj_name="neurips_test_f2f")

    # -- misc --
    # import pickle
    # fn = ".cache_io_pkl/trte_f2f/test.pkl"
    # with open(fn,"rb") as f:
    #     results = pickle.load(f)

    # -- view --
    results = results.fillna("None")
    # results = results[results['nepochs'] == 5]
    results = results.rename(columns={"gradient_clip_val":"gcv"})

    # -- confirm testing --
    for tr_uuid,gdf in results.groupby(["tr_uuid","dname","sigma","rate"]):
        print(tr_uuid,len(gdf))

    print(len(results))
    if len(results) == 0: return
    afields = ['psnrs','ssims','strred','psnrs_pp','ssims_pp','strred_pp']
    # gfields = ["search_input","crit_name","dist_crit","stride0",
    #            "ps","k","ws","wt","gcv","dset_tr","sigma",'nepochs']
    # gfields = ["dist_crit","k","ws","ps","nepochs","ps_dists","stride0",
    #            "iphone_type"]
    # gfields = ["crit_name","nepochs","iphone_type"]
    gfields = ["crit_name","nepochs","dname",'sigma','rate',"tr_uuid"]
    # optional = ["iphone_type","stnls_center_crop","label0"]
    optional = ["stnls_center_crop","label0"]
    for opt in optional:
        if opt in results.columns:
            gfields += [opt,]
    print(gfields)
    print(len(results),len(results[gfields].drop_duplicates()))
    # results = results[results['vid_name'] == "sunflower"].reset_index(drop=True)
    print(results)
    results = results.sort_values("nepochs")
    print(len(results))
    agg_fxn = lambda x: np.mean(np.stack(x))
    for f in afields: results[f] = results[f].apply(np.mean)
    summary = results.groupby(gfields).agg({k:agg_fxn for k in afields})
    summary = summary.reset_index()[gfields + afields]
    # summary = summary[summary['nepochs'] == 30]
    print(summary)

    # -- uniques --
    print(summary['dname'].unique())
    print(summary['sigma'].unique())
    print(summary['rate'].unique())


    # -- split groups --
    # key = 'ssims'
    # gfields = ["crit_name","nepochs",]
    gfields = ["crit_name","nepochs","dname",'sigma','rate',"tr_uuid"]
    optional = ["stnls_center_crop"]
    # optional = ["iphone_type","stnls_center_crop"]
    for opt in optional:
        if opt in results.columns:
            gfields += [opt,]


    key = ['psnrs','ssims','strred',]#'psnrs_pp','ssims_pp','strred_pp']
    key_pp = ['psnrs_pp','ssims_pp','strred_pp']
    # gfields = ["crit_name","nepochs"]
    for group0,gdf0 in summary.groupby(["dname","sigma","rate"]):
        print(group0)
        for group1,gdf1 in gdf0.groupby(["label0","tr_uuid"]):
            label0 = group1[0]
            for group,gdf in gdf1.groupby(gfields):
                if "b2u" in label0:
                    print(f"{group1}: ",gdf[key+key_pp].to_numpy())#," : ",group)
                else:
                    print(f"{group1}: ",gdf[key].to_numpy())#," : ",group)

if __name__ == "__main__":
    main()
