"""

Produces weights for denoising models.

"""

# -- sys --
import os

# -- caching results --
import cache_io

# -- network configs --
# from dnls_paper import train_model
from dnls_paper import reports

# -- bench --
from dev_basics.trte import train

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- records --
    def clear_fxn(num,cfg):
        return False
    exps,uuids = cache_io.train_stages.run("exps/trte_colanet/train.cfg",
                                           ".cache_io/trte_colanet/train/")
    # -- filter --
    exps_,uuids_ = exps,uuids
    exps,uuids = [],[]
    for e,u in zip(exps_,uuids_):
        if e.wt == 3 and e.read_flow == True and e.sigma == 50:
            exps.append(e)
            uuids.append(u)
    print(len(exps))
    records = cache_io.run_exps(exps,train.run,uuids=uuids,
                                name=".cache_io/trte_colanet/train",
                                enable_dispatch="slurm")
    # -- view --
    print(records.head())


if __name__ == "__main__":
    main()
