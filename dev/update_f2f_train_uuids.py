
"""

A bug in cache_io means we must map the distributed 
train cache_io uuids to the primary cache_io

"""

import cache_io
import tqdm
from pathlib import Path

def update_uuid(base,dist):

    # -- read valid uuids --
    dist_data = dist.uuid_cache.data
    cfgs,uuids = [],[]
    for i in range(len(dist_data['config'])):
        uuid = dist_data['uuid'][i]
        print(uuid)
        path = Path("output/train/trte_f2f/checkpoints/") / uuid
        if not(path.exists()): continue
        print("exist.")
        cfgs.append(dist_data['config'][i])
        uuids.append(dist_data['uuid'][i])
    # print(len(cfgs))

    # -- ensure only one --
    assert len(cfgs) == 1
    cfg,uuid = cfgs[0],uuids[0]

    # -- translate --
    # for key in cfg:
    #     condA = "f2f" in str(cfg[key])
    #     condB = "dnls" in str(cfg[key])
    #     if not(condA or condB): continue
    #     og = cfg[key]
    #     # cfg[key] = cfg[key].replace("f2f","trte_f2f")
    #     cfg[key] = cfg[key].replace("dnls","stnls")
    #     print(key,og,cfg[key])
    # print(cfg)

    # -- ensure only one --
    base_uuid = base.read_uuid(cfg)
    assert base_uuid != -1
    return uuid,cfg

def main():

    # -- config --
    dist_fmt = ".cache_io/tr_f2f_dispatch_%d"
    base = cache_io.ExpCache(".cache_io/trte_f2f/train")
    num = 29

    # -- read uuids --
    uuids,exps = [],[]
    for i in tqdm.tqdm(range(num)):
        dist = cache_io.ExpCache(dist_fmt % i)
        uuid_i,exp_i = update_uuid(base,dist)
        uuids.append(uuid_i)
        exps.append(exp_i)

    # -- save --
    print(uuids[:5])
    base.uuid_cache.write_pair(uuids,exps)
    base_exp = cache_io.ExpCache(".cache_io_exps/trte_f2f/train")
    base_exp.uuid_cache.write_pair(uuids,exps)


def update_cache_io_exps():

    # -- update exps --
    base = cache_io.ExpCache(".cache_io/trte_f2f/train")
    base_exp = cache_io.ExpCache(".cache_io_exps/trte_f2f/train")
    base_exp.uuid_cache.write(base.uuid_cache.data)


if __name__ == "__main__":
    main()
    # update_cache_io_exps()
