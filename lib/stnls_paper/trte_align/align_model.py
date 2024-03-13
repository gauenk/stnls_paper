
import torch
import torch as th
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import copy
dcopy = copy.deepcopy
from einops import rearrange

from dev_basics import flow as flow_pkg
import stnls
from .spynet import SpyNet
from ..gda import load_model as init_gda
from stnls.search.paired_search import init as init_paired_search
from stnls.search.paired_refine import init as init_paired_refine

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
        self.ws = cfg.ws
        self.wt = 1
        self.stride0 = cfg.stride0
        self.align_type = align_type
        self.spynet = [SpyNet(load_path=load_path).cuda()]
        if self.align_type == "gda":
            self.model = init_gda(cfg)
        elif self.align_type == "stnls":
            self.model = init_paired_search(cfg)
            self.refine = None
        else:
            raise ValueError(f"Uknown align [{self.align_type}]")

        # -- legalize boundaries --
        cfg = dcopy(cfg)
        cfg.ws = 1
        cfg.wr = 1
        cfg.ps = 1
        self.bounds = init_paired_search(cfg)

    def forward(self,vid,flows=None):

        # -- compute optical flows --
        # fflow,bflow = self.spynet[0].compute_flow(vid)
        # print(fflow.shape,bflow.shape)
        # wt = vid.shape[1]-1

        # -- compute optical flows --
        if flows is None:
            flows = flow_pkg.orun(vid,True,ftype="cv2")
            fflow,bflow = flows.fflow,flows.bflow
            flows = stnls.nn.search_flow(fflow,bflow,self.wt,self.stride0)

        # -- spynet ? --
        # flows = flow.orun(nvid,cfg.flow,ftype="cv2")
        # flow_norm = (flows.fflow.abs().mean() + flows.bflow.abs().mean()).item()/2.
        # # acc_flows = stnls.nn.accumulate_flow(flows.fflow,flows.bflow)
        # acc_flows = stnls.nn.search_flow(flows.fflow,flows.bflow,cfg.wt,cfg.stride0)
        # flows = stnls.nn.accumulate_flow(fflow,bflow,self.stride0)

        # -- correct the boundary --
        if self.bounds:
            flows_k = self.bounds.paired_vids(vid, vid, flows,
                                              self.wt, skip_self=True)[1]
            flows_k = flows_k[...,1:].flip(-1)
            flows = rearrange(flows_k,'b 1 t h w wt two -> b t wt two h w')

        # -- model --
        flows_k = self.model.paired_vids(vid, vid, flows, self.wt, skip_self=True)[1]
        if self.align_type == "stnls":
            flows_k[...,1:] = flows_k[...,1:].flip(-1)

        # -- allow learning the identity function [stnls wraps boundarys] --
        # if (self.align_type == "stnls") and (self.ws == 1):
        #     time = flows_k[...,[0]]
        #     _flows = rearrange(flows,'b t a tw h w -> b 1 t h w a tw')
        #     flows_k = th.cat([time,_flows],-1)

        #         if self.align_type == "gda":
        # (flows_k)
        # flows_k.shape = (b,hd,t,H,W,k*wt,3)
        # flows_k[...,0] = time, flows_k[...,1:] = (height,width)
        # print("flows_k.shape: ",flows_k.shape)
        # # print(flows_k[0,0,0,5,5])
        # # print(flows_k[0,0,0,60,60])
        # exit()

        # if self.align_type == "stnls" or True:
        #     print("align_type: ",self.align_type)
        #     print("[align_model] flows.shape: ",flows.shape)
        #     print("[align_model] flows_k.shape: ",flows_k.shape)
        #     print("-"*10)
        #     print(flows[0,0,:,:,0,0])
        #     print(flows_k[0,0,0,0,0,:,1:])
        #     print("*"*10)
        #     # print("-"*20)
        #     # h,w = 16,16
        #     # print(flows[0,0,:,:,h,w].shape)
        #     # print(flows_k[0,0,0,h,w,:,1:].shape)
        #     # print("-"*10)
        #     # print(flows[0,0,:,:,h,w])
        #     # print(flows_k[0,0,0,h,w,:,1:])
        #     # print("*"*10)
        #     if self.align_type == "stnls":
        #         exit()

        return flows_k

