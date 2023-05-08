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
    exps,uuids = cache_io.train_stages.run("exps/trte_colanet/train.cfg",
                                           ".cache_io_exps/trte_colanet/train/")
                                           # update=True)

    # print(uuids)
    print("Num Exps: ",len(exps))
    results = cache_io.run_exps(exps,train.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_colanet/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_colanet/train.pkl",
                                records_reload=False,
                                use_wandb=True,proj_name="neurips_colanet")
    # -- view --
    if len(results) == 0: return
    print(results.head())


if __name__ == "__main__":
    main()
