"""

Produces weights for denoising models.

"""

# -- sys --
import os

# -- caching results --
import cache_io

# -- network configs --
from icml23 import train_model
from icml23 import reports

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- records --
    cfg_file = "exps/trte_n3net/finetune_n3net.cfg"
    exps = cache_io.get_exps(cfg_file)
    records = cache_io.run_exps(exps,train_model.run,
                                enable_dispatch="slurm")
    print(records.head())


if __name__ == "__main__":
    main()
