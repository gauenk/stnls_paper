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

    # -- records --
    cfg_file = "exps/trte_colanet/test.cfg"
    records = cache_io.run_exps(cfg_file,test.run,
                                # name="trte_colanet/test",
                                # name=".cache_io/test_colanet_01_04/",
                                name=".cache_io/test_colanet_01_02/",
                                skip_loop=False)

    # -- print table information --
    print(records)

if __name__ == "__main__":
    main()
