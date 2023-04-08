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
    def clear_fxn(num,cfg):
        return False
    read_test = cache_io.read_test_config.run
    exps = read_test("exps/trte_colanet/test.cfg",
                     cache_name=".cache_io_exps/trte_colanet/test",reset=True)
    exps,uuids = cache_io.get_uuids(exps,".cache_io/trte_colanet/test",
                                    reset=True,no_config_check=True)

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,
                                name=".cache_io/trte_colanet/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_colanet/test.pkl",
                                records_reload=False,to_records_fast=True)

    # -- print table information --
    print(records)

if __name__ == "__main__":
    main()
