"""

Script to train a model

"""

import os,pprint
pp = pprint.PrettyPrinter(indent=4)
import cache_io
from icml23 import train_model
# from dev_basics import exps as dev_exps
from cache_io import exps as dev_exps

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "train_nets"
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- load exp --
    # fn = "exps/finetune_stardeno_wt.cfg"
    # fn = "exps/finetune_stardeno.cfg"
    fn = "exps/finetune_colanet_rbwd.cfg"
    exps = dev_exps.load(fn)
    # exp = exps[0]
    exp = exps[1]

    # fn = "exps/finetune_lidia.cfg"
    # exps = dev_exps.load(fn)
    # exp = exps[1]

    # fn = "exps/finetune_n3net.cfg"
    # exps = dev_exps.load(fn)
    # exp = exps[0]

    # -- add uuid --
    pp.pprint(exp)
    uuid = cache.get_uuid(exp)
    exp.uuid = uuid

    # -- run --
    train_model.run(exp)


if __name__ == "__main__":
    main()
