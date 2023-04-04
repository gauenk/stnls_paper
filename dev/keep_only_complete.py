
"""

A bug in cache_io means we must map the distributed 
train cache_io uuids to the primary cache_io

"""

import cache_io
import tqdm
from pathlib import Path

def main():

    # -- config --
    base_path = Path(".cache_io/trte_tinyvrt/test/")
    base = cache_io.ExpCache(str(base_path))
    data = base.uuid_cache.data

    # -- read uuids --
    uuids,cfgs = [],[]
    uuids_,cfgs_ = data['uuid'],data['config']
    num = 0
    for i in range(len(uuids_)):
        path_i = base_path / uuids_[i]
        if not(path_i.exists()): continue
        num += 1
        uuids.append(uuids_[i])
        cfgs.append(cfgs_[i])
    # print(num,len(uuids),len(cfgs))

    # -- save --
    # base.uuid_cache.write_pair(uuids,cfgs)

if __name__ == "__main__":
    main()
