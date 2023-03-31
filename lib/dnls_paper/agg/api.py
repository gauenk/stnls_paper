import n3net
from easydict import EasyDict as edict

def init_agg(cfg,name):
    searches = edict()
    modules = {"n3net":n3net}
    stypes = ["original","ours"]
    mod = modules[name]
    for stype in stypes:
        # name_s = "%s_%s" % (name,stype)
        name_s = stype
        search_cfg = mod.extract_agg_config(cfg)
        search_cfg.name = stype
        searches[name_s] = mod.get_agg(search_cfg)
    return searches
