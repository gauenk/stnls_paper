"""

Compare the impact of train/test using exact/refineimate methods


"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from dev_basics.trte import train,bench

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get experiments --
    def clear_fxn(num,cfg): return False
    # read_filter = {"read_flows":True,"num_res":10,"nres_per_block":10,
    #                "nepochs":300,"input_proj_depth":[1],
    #                "save_epoch_list":"1-50-100-150-200-250"}
    # read_filter = None
    exps,uuids = cache_io.train_stages.run("exps/trte_nlnet/train.cfg",
                                           ".cache_io_exps/trte_nlnet/train/",update=True)
    print(uuids)

    # -- view --
    # bench.print_summary(exps[0],(1,3,3,128,128))
    # bench.print_summary(exps[0],(1,3,3,512,512))
    results = []
    # vshape = (1,10,3,512,512)
    # vshape = (1,5,3,360,360)
    # vshape = (2,6,3,256,256)
    # vshape = (4,4,3,256,256)
    vshape = (1,10,3,256,256)
    # vshape = (1,10,3,280,280)
    n = 0
    for exp in exps:
        res = bench.summary(exp,vshape,with_flows=True)
        res.arch_depth = exp.arch_depth
        res.arch_nheads = exp.arch_nheads
        res.embed_dim = exp.embed_dim
        res.qk_frac = exp.qk_frac
        res.num_res = exp.num_res
        res.nres_per_block = exp.nres_per_block
        results.append(res)
        # if n > 5: break
        n+=1
    results = pd.DataFrame(results)
    print(results[['arch_depth','arch_nheads','embed_dim','qk_frac','num_res','nres_per_block']])
    print(results[['timer_fwd','trainable_params',"macs"]])
    print(results[['timer_fwd','timer_bwd']])
    print(results[['alloc_fwd','alloc_bwd','res_fwd','res_bwd']])
    print(results[['fwdbwd_mem','total_params','macs']])


if __name__ == "__main__":
    main()

