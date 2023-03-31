"""

Produces weights for denoising models.

"""

# -- sys --
import os

# -- caching results --
import cache_io

# -- network configs --
from dnls_paper import train_model
from dnls_paper import reports

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- records --
    cfg_file = "exps/finetune_colanet.cfg"
    records = cache_io.run_exps(cfg_file,train_model.run,
                                enable_dispatch="slurm")
    print(records.head())


if __name__ == "__main__":
    main()
