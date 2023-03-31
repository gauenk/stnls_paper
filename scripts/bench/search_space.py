"""

Grid search over the window sizes of the search space

"""


# -- sys --
import os

# -- caching results --
import cache_io

# -- data mangling --
import numpy as np

# -- network configs --
from icml23 import test_model
from icml23 import plots

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- run experiments --
    exps = ["exps/search_space_colanet.cfg",
            "exps/search_space_n3net.cfg",
            "exps/search_space_lidia.cfg"]
    records = cache_io.run_exps(exps,test_model.run,
                                name=".cache_io/search_space",
                                version="v1",
                                records_fn =".search_space.pkl",
                                records_reload=False,
                                skip_loop=True)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    # --         Plot          --
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-

    fields = ['psnrs','strred','mem','time']
    field_labels = [r"PSNR (dB) $\uparrow$",r"ST-RRED ($10^{-2}$) $\downarrow$",
                    "Memory (GB)","Runtime (sec)"]

    # -- across architectures --
    for arch,adf in records.groupby("arch_name"):

        # -- plot across fields --
        for field,field_label in zip(fields,field_labels):
            plots.search_space.run(adf,field,field_label)


if __name__ == "__main__":
    main()
