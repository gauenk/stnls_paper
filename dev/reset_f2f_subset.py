import cache_io
from pathlib import Path
import shutil

def main():
    exps,uuids = cache_io.train_stages.run("exps/trte_f2f/train.cfg",
                                           ".cache_io_exps/trte_f2f/train/")
    for cfg,uuid in zip(exps,uuids):
        clear_bool = "crit_name" in cfg and cfg["crit_name"] == "stnls"
        if clear_bool:
            path = Path("output/train/trte_f2f/checkpoints/%s" % uuid)
            if path.exists(): shutil.rmtree(str(path))
    # cache = cache_io.ExpCache(".cache_io/trte_f2f/train")
    # results = cache_io.run_exps(exps,train.run,uuids=uuids,
    #                             name=".cache_io/trte_f2f/train",
    #                             version="v1",skip_loop=False,clear_fxn=clear_fxn,
    #                             clear=False,enable_dispatch="slurm",
    #                             records_fn=".cache_io_pkl/trte_f2f/train.pkl",
    #                             records_reload=False)


if __name__ == "__main__":
    main()
