"""

Keep only the completed cache

"""

import numpy as np
import cache_io
from pathlib import Path


def main():

    # -=-=-=-=-=-=-=-=-=-
    #
    #      Dispatch
    #
    # -=-=-=-=-=-=-=-=-=-

    base_fmt = ".cache_io/te_f2f_dispatch_%d"
    skip = 760
    num = 21128
    missing = {}
    count_path = 0
    count_cache = 0
    for i in range(0,num,skip):

        # -- uuids in directory --
        path_s = base_fmt % i
        path = Path(path_s)
        path_uuids = [str(p.name) for p in path.iterdir() if "-" in str(p)]
        path_uuids = sorted(path_uuids)
        # print(len(path_uuids))
        count_path += len(path_uuids)

        # -- uuids in cache --
        exp = cache_io.ExpCache(path_s)
        cache_uuids = exp.uuid_cache.data['uuid']
        count_cache += len(cache_uuids)

        # -- skip if equal len --
        # if len(path_uuids) == len(cache_uuids): continue

        # -- ensure no path DNE in cache --
        for p_uuid in path_uuids:
            assert p_uuid in cache_uuids

        # -- find num cache DNE in path --
        for c_uuid in cache_uuids:
            if c_uuid in path_uuids: continue
            if not(path_s in missing):
                missing[path_s] = []
            missing[path_s].append(c_uuid)

        # -- remove missing --
        # if not(path_s in missing): continue
        # data = exp.uuid_cache.data
        # cfgs,uuids = list(data['config']),list(data['uuid'])
        # for missing_uuid in missing[path_s]:
        #     index = uuids.index(missing_uuid)
        #     del uuids[index]
        #     del cfgs[index]
        # exp.uuid_cache.write_pair(uuids,cfgs)

    # -- view --
    print(missing)
    print(count_path)
    print(count_cache)

    # -=-=-=-=-=-=-=-=-=-
    #
    #     Base Cache
    #
    # -=-=-=-=-=-=-=-=-=-

    # -- init --
    missing = {}

    # -- uuids in directory --
    path_s = ".cache_io/f2f/test/"
    path = Path(path_s)
    path_uuids = [str(p.name) for p in path.iterdir() if "-" in str(p)]
    path_uuids = sorted(path_uuids)

    # -- uuids in cache --
    exp = cache_io.ExpCache(path_s)
    cache_uuids = exp.uuid_cache.data['uuid']

    # -- skip if equal len --
    # if len(path_uuids) == len(cache_uuids): continue

    # -- ensure no path DNE in cache --
    for p_uuid in path_uuids:
        assert p_uuid in cache_uuids

    # -- find num cache DNE in path --
    for c_uuid in cache_uuids:
        if c_uuid in path_uuids: continue
        if not(path_s in missing):
            missing[path_s] = []
        missing[path_s].append(c_uuid)

    print(missing)


if __name__ == "__main__":
    main()
