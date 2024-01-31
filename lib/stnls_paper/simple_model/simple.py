"""

   Simple Network

"""

import torch
import torch as th
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from ..gda import GdaForVideoAlignment
from .spynet import SpyNet


def optional(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default

def extract_defaults(_cfg):
    cfg = edict(dcopy(_cfg))
    defs = {"dim":9,"spynet_load_path":"",  "conv_ksize":3,
            "block_num":1, "attn_type":"gda",
            "qk_dim":9, "qk_scale":1., "mlp_dim":9, "topk":10,
            "dist_type":"l2", "use_weights":True,}
    for k in defs: cfg[k] = optional(cfg,k,defs[k])
    return cfg

def load_model(cfg):
    model = SimpleModel(cfg.dim, cfg.spynet_load_path,
                        cfg.attn_type,
    )
    return model

class SimpleModel():

    def __init__(self, dim, spynet_load_path, conv_ksize, block_num,
                 attn_type, heads, qk_dim, qk_scale, mlp_dim, topk,
                 dist_type, use_weights):
        super(self).__init__()
        self.attn_type = attn_type
        self.spynet = SpyNet(load_path=spynet_load_path)
        self.first_layer = nn.Conv2d(3, dim, 3, 1, 1)
        self.last_layer = nn.Conv2d(dim, 3, 3, 1, 1)

        self.blocks = []
        for i in range(block_num):
            self.blocks.append(Block(dim=dim, attn_type=attn_type,
                                     heads=heads, qk_dim=qk_dim,
                                     qk_scale=qk_scale,
                                     mlp_dim=mlp_dim,topk=topk,
                                     dist_type=dist_type,
                                     use_weights=use_weights))
            if self.use_midconvs:
                self.mid_convs.append(nn.Conv2d(dim, dim, 3, 1, 1))
            else:
                self.mid_convs.append(nn.Identity())

    def forward(self,vid):

        # -- get flows --
        fflow,bflow = self.spynet.compute_first_order_flows(vid)

        # -- input projection --
        vid = self.first_layer(vid)

        # -- bulk of layers --
        for layer in self.layer_list:
            vid = self.subclip_fwd(layer,vid,fflow,bflow)

        # -- final projection --
        vid = self.last_layer(vid)

        return vid

    def subclip_fwd(self,layer,vid,fflow,bflow):

        # -- allocate --
        B,T,F,H,W = vid.shape
        outs = th.zeros_like(vid)
        Z = th.zeros((1,T,1,1,1),device=vid.device,dtype=th.int)

        # -- apply each subclip --
        for ti in range(T-1):

            # -- process pair fwd and bwd --
            subclip_f = layer(vid[:,ti],vid[:,ti+1],fflow[:,ti])
            subclip_b = layer(vid[:,ti+1],vid[:,ti],bflow[:,ti+1])

            # -- accumulate --
            outs[:,ti] += subclip_f
            outs[:,ti+1] += subclip_b
            Z[:,ti] += 1
            Z[:,ti+1] += 1

        # -- normalize --
        outs = outs / Z

        return outs
