
"""

A bug in cache_io means we must map the distributed
train cache_io uuids to the primary cache_io

"""

import cache_io
import tqdm
from pathlib import Path

def main():

    # -- config --
    base = cache_io.ExpCache(".cache_io/trte_f2f/train")
    chkpt_base = Path("output/train/trte_f2f/checkpoints/")
    uuids,cfgs = [],[]
    for path_uuid in chkpt_base.iterdir():
        uuid = path_uuid.name
        cfg = base.uuid_cache.get_config_from_uuid(uuid)
        uuids.append(uuid)
        cfgs.append(cfg)
    base.uuid_cache.write_pair(uuids,cfgs)

if __name__ == "__main__":
    main()
