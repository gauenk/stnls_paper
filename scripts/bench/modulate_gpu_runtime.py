"""

Show modulation of GPU Memory and Runtime

"""


# -- misc --
import os

# -- cache --
import cache_io

# -- network configs --
from icml23 import test_model
from icml23 import plots

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run cached records --
    cfg_file = "exps/modulate_gpu_runtime.cfg"
    records = cache_io.run_exps(cfg_file,test_model.run,
                                name=".cache_io/modulate_gpu_runtime",
                                version="v1",skip_loop=True)

    # -- plots --
    plots.modulate_gpu_runtime.run(records)

if __name__ == "__main__":
    main()
