import colanet
import lidia
import n3net
from easydict import EasyDict as edict

def init_search(cfg,mod_name,search_name):
    searches = edict()
    modules = {"colanet":colanet,
               "lidia":lidia,"n3net":n3net}
    module = modules[mod_name]
    search_cfg = module.extract_search_config(cfg)
    search_cfg.name = search_name
    search_fxn = module.get_search(search_cfg)
    return search_fxn
