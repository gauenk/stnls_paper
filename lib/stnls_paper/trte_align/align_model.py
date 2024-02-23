
import torch
import torch as th
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from dev_basics import flow as flow_pkg
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
        self.spynet = [SpyNet(load_path=load_path).cuda()]
        if self.align_type == "gda":
            self.model = init_gda(cfg)
        elif self.align_type == "stnls":
            self.model = init_paired_search(cfg)
        else:
            raise ValueError(f"Uknown align [{self.align_type}]")

    def forward(self,vid):

        # -- compute optical flows --
        # fflow,bflow = self.spynet[0].compute_flow(vid)
        # print(fflow.shape,bflow.shape)
        # wt = vid.shape[1]-1

        flows = flow_pkg.orun(vid,True,ftype="cv2")
        fflow,bflow = flows.fflow,flows.bflow

        flows = stnls.nn.search_flow(fflow,bflow,self.wt,self.stride0)

        # flows = flow.orun(nvid,cfg.flow,ftype="cv2")
        # flow_norm = (flows.fflow.abs().mean() + flows.bflow.abs().mean()).item()/2.
        # # acc_flows = stnls.nn.accumulate_flow(flows.fflow,flows.bflow)
        # acc_flows = stnls.nn.search_flow(flows.fflow,flows.bflow,cfg.wt,cfg.stride0)

        # flows = stnls.nn.accumulate_flow(fflow,bflow,self.stride0)
        # print(vid.min(),vid.max())
        # print(flows.shape)
        # print(fflow.min(),fflow.max())
        # print(bflow.min(),bflow.max())
        # print(flows.min(),flows.max())
        # exit()

        # -- compute topk flows [aka the "corrections"] --
        # print("\tvid.shape: ",vid.shape)
        # print("\tflows.shape: ",flows.shape)
        flows_k = self.model.paired_vids(vid, vid, flows, self.wt, skip_self=True)[1]
        # print("flows_k.shape: ",flows_k.shape)
        # # print(flows_k[0,0,0,5,5])
        # # print(flows_k[0,0,0,60,60])
        # exit()

        return flows_k

