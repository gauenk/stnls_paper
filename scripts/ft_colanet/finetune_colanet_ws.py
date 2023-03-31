"""

Test the finetuned n3net to show the differences

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
    cfg_file = "exps/finetune_colanet_ws.cfg"
    records = cache_io.run_exps(cfg_file,train_model.run,
                                enable_dispatch="slurm")
    print(records.head())


if __name__ == "__main__":
    main()
