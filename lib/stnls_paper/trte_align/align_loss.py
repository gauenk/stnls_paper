

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
        cfg.k = _cfg.k
        cfg.ws = _cfg.ws
        cfg.wt = 1
        cfg.kr = 1.
        cfg.ps = _cfg.ps
        cfg.nheads = 1
        cfg.dilation = 1
        cfg.stride0 = 1
        cfg.stride1 = 1
        cfg.reflect_bounds=True
        cfg.self_action=None
        cfg.dist_type = "l2"
        cfg.itype="float"
        self.ws = cfg.ws
        self.gt_model = AlignModel(cfg,"stnls",_cfg.spynet_path)
        self.loss_fxn_input = _cfg.loss_fxn_input
        # self.refine = stnls.search.RefineSearch(ws, wt, 1, -1, kr, ps, nheads,
        #                                         dilation=dil,stride0=stride0,
        #                                         stride1=stride1,full_ws=True,
        #                                         reflect_bounds=reflect_bounds,
        #                                         self_action=self_action,
        #                                         dist_type=dist_type,itype=itype)

    def forward(self,vid,noisy,flow):
        # print(flow)
        # flow =rearrange(flow,'b hd k t tr h w -> b hd t h w k tr')
        # print("vid.shape, flow.shape: ",vid.shape, flow.shape)
        # print(flow[0,0,0,5,5])
        # print(flow[0,0,0,10,10])

        ins = None
        if self.loss_fxn_input == "clean":
            ins = vid
        elif self.loss_fxn_input == "noisy":
            ins = noisy
        else:
            raise ValueError(f"Uknown loss function [{self.loss_fxn_input}]")
        assert not(ins == None)

        flows_gt = self.gt_model(ins)
        # dists,flows_r = self.refine(vid,vid,flow)
        # print(dists)
        # print(dists.min())
        # print(dists.max())
        # print(flow.min(),flow.max())
        # print(flows_r.min(),flows_r.max())
        # exit()
        # return th.mean(dists)
        # print(flows_gt.shape)
        # print("flow.shape: ",flow.shape)
        # print("flows_gt.shape: ",flows_gt.shape)
        # print("-"*10)
        # print(flow[0,0,0,5:8,5:8,:,1])
        # print(flows_gt[0,0,0,5:8,5:8,:,1])
        # print("-"*10)
        # print(flow[0,0,:,2,2,:,1])
        # print(flows_gt[0,0,:,2,2,:,1])
        # print("-"*10)
        # print(flow[0,0,:,2,2,:,2])
        # print(flows_gt[0,0,:,2,2,:,2])
        # exit()

        return th.mean((flows_gt - flow)**2)
