"""

Produces weights for denoising models.

"""

# -- sys --
import os

# -- caching results --
import cache_io

# -- network configs --
# from stnls_paper import train_model
from stnls_paper import reports

# -- bench --
from dev_basics.trte import train

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- records --
    def clear_fxn(num,cfg): return False
    exps,uuids = cache_io.train_stages.run("exps/trte_sr/train.cfg",
                                           ".cache_io/trte_sr/train/",update=True)
    print("Num Exps: ",len(exps))
    results = cache_io.run_exps(exps,train.run,uuids=uuids,
                                name=".cache_io/trte_sr/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_sr/train.pkl",
                                records_reload=False)
    # -- view --
    if len(results) == 0: return
    print(results.head())


if __name__ == "__main__":
    main()

