import torch
import torch as th
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .gda_attn import GDA
from stnls.nn import NonLocalAttentionStack

class Block(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        num (int): Number of blocks.
        heads (int): Head numbers of Attention.
        qk_dim (int): Channels of query and key in Attention.
        mlp_dim (int): Channels of hidden mlp in FFN.
    """
    def __init__(self, dim=9, attn_type="mask",
                 heads=1, qk_dim=9, qk_scale=1.,
                 mlp_dim=9,topk=10, dist_type="l2",
                 use_weights=True):
        super(Block,self).__init__()

        # -- assign -
        self.dim = dim
        self.attn_type = attn_type
        self.heads = heads
        self.qk_dim = qk_dim
        self.qk_scale = qk_scale
        self.mlp_dim = mlp_dim
        self.topk = topk
        self.dist_type = dist_type
        self.use_weights = use_weights

        # -- init attn --
        if self.attn_type =="gda":
            self.attn = GDA()
        elif self.attn_type == "stnls":
            self.attn = NonLocalAttentionStack()


    def forward(self, frame0, frame1, flows):

        frames = th.cat([frame0,frame1],-3) # across channels
        frames = self.norm(frames)
        frames = self.conv_block(frames)


        return frames
