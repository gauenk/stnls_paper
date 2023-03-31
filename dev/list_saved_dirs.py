"""

List the directory of saved output files so it can be copied to host machine

"""

import os
import cache_io
import pandas as pd
from pathlib import Path

def get_row_files(row):
    path = Path(row['saved_dir']) / row['arch_name'] / row['uuid']
    if path.exists(): return path
    else: return ""
    return path

def get_files(records):
    paths = []
    for _,row in records.iterrows():
        path = get_row_files(row)
        if path != "":
            path = path.relative_to("/home/gauenk/Documents/packages/aaai23")
            paths.append(path)
    return paths

def save_paths(paths,fn):
    paths = pd.DataFrame({"paths":paths})
    paths.to_csv(fn,index=False,header=False)

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- run/load exps --
    exp_files = ["exps/test_colanet.cfg"]
    exps = cache_io.get_exps(exp_files)
    records = cache_io.run_exps(exps,None,
                                name=".cache_io/test_colanet_01_04",
                                version="v1",skip_loop=True,
                                records_fn=".cache_io_pkl/test_colanet_01_04.pkl",
                                records_reload=False)
    print(len(records),len(exps))

    # -- enumerate exisiting files --
    paths = get_files(records)
    save_paths(paths,"existing_paths.txt")


if __name__ == "__main__":
    main()
