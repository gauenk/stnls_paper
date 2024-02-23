

import torch
import torch as th
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import stnls
from einops import rearrange
from easydict import EasyDict as edict
from .align_model import AlignModel

class AlignLoss(nn.Module):

    def __init__(self,_cfg):
        super().__init__()
        cfg = edict()
        cfg.k = 1
        cfg.ws = 10
        cfg.wt = 1
        cfg.kr = 1.
        cfg.ps = 1
        cfg.nheads = 1
        cfg.dilation = 1
        cfg.stride0 = 1
        cfg.stride1 = 1
        cfg.reflect_bounds=True
        cfg.self_action=None
        cfg.dist_type = "l2"
        cfg.itype="float"
        self.gt_model = AlignModel(cfg,"stnls",_cfg.spynet_path)
        # self.refine = stnls.search.RefineSearch(ws, wt, 1, -1, kr, ps, nheads,
        #                                         dilation=dil,stride0=stride0,
        #                                         stride1=stride1,full_ws=True,
        #                                         reflect_bounds=reflect_bounds,
        #                                         self_action=self_action,
        #                                         dist_type=dist_type,itype=itype)

    def forward(self,vid,flow):
        # print(flow)
        # flow =rearrange(flow,'b hd k t tr h w -> b hd t h w k tr')
        # print("vid.shape, flow.shape: ",vid.shape, flow.shape)
        # print(flow[0,0,0,5,5])
        # print(flow[0,0,0,10,10])
        flows_gt = self.gt_model(vid)
        # print("flow.shape,flows_gt.shape: ",flow.shape,flows_gt.shape)
        # dists,flows_r = self.refine(vid,vid,flow)
        # print(dists)
        # print(dists.min())
        # print(dists.max())
        # print(flow.min(),flow.max())
        # print(flows_r.min(),flows_r.max())
        # exit()
        # return th.mean(dists)
        return th.mean((flows_gt - flow)**2)
