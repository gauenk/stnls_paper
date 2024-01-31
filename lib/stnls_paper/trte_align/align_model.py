
import torch
import torch as th
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import stnls
from .spynet import SpyNet
from ..gda import load_model as init_gda
from stnls.search.paired_search import init as init_paired_search

# def extract_config(cfg,restrict=True):
#     pairs = {"ws":-1,"ps":3,"k":10,
#              "nheads":1,"dist_type":"l2",
#              "stride0":1, "stride1":1, "dilation":1, "pt":1,
#              "reflect_bounds":True, "full_ws":True,
#              "self_action":None,"use_adj":False,
#              "normalize_bwd": False, "k_agg":-1,
#              "off_Hq":0,"off_Wq":0,"itype":"float",}
#     return extract_pairs(cfg,pairs,restrict=restrict)

class AlignModel(nn.Module):

    def __init__(self, cfg, align_type, load_path):
        super().__init__()
        self.wt = 1
        self.stride0 = cfg.stride0
        self.align_type = align_type
        self.spynet = SpyNet(load_path=load_path)
        if self.align_type == "gda":
            self.model = init_gda(cfg)
        elif self.align_type == "stnls":
            self.model = init_paired_search(cfg)
        else:
            raise ValueError(f"Uknown align [{self.align_type}]")

    def forward(self,vid):

        # -- compute optical flows --
        fflow,bflow = self.spynet.compute_flow(vid)
        print(fflow.shape,bflow.shape)
        flows = stnls.nn.search_flow(fflow,bflow,self.wt,self.stride0)

        # -- compute topk flows [aka the "corrections"] --
        flows_k = self.model.paired_vids(vid, vid, flows, self.wt, skip_self=False)[1]
        print("flows_k.shape: ",flows_k.shape)

        return flows_k

