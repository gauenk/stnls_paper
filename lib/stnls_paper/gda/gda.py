"""

GDA for video alignment

"""

import torch
import torch as th
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from stnls.search.utils import paired_vids as _paired_vids
from einops import rearrange

def load_model(cfg):
    model = GdaForVideoAlignment(attn_size=cfg.k)#attn_size)
    return model

class GdaForVideoAlignment(nn.Module):
    """Guided deformable attention module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        attention_window (int or tuple[int]): Attention window size. Default: [3, 3].
        attention_heads (int): Attention head number.  Default: 12.
        deformable_groups (int): Deformable offset groups.  Default: 12.
        clip_size (int): clip size. Default: 2.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
    Ref:
        Recurrent Video Restoration Transformer with Guided Deformable Attention

    """

    def __init__(self, in_channels=3, deformable_groups=1, attn_size=9, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.clip_size = 1
        self.attn_size = attn_size
        self.deformable_groups = deformable_groups
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 0.01)
        self.fixed_offset_max = kwargs.pop('fixed_offset_max', 2.5)

        num_in = self.in_channels * (1 + self.clip_size) + self.clip_size * 2
        # print(num_in)
        self.conv_offset = nn.Sequential(
            nn.Conv3d(num_in, 64, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, self.clip_size * self.deformable_groups *\
                      self.attn_size * 2, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0)),
        )

    def forward(self,frame0,frame1,flow):
        # print("[gda] flow.shape: ",flow.shape)
        warped_frame0 = flow_warp(frame0,flow[:,0])
        frame0,frame1 = frame0[:,None],frame1[:,None]
        warped_frame0 = warped_frame0[:,None]
        # flow = flow.flip(2)
        flows_k = self.conv_offset(torch.cat([frame1,warped_frame0,flow], 2)\
                               .transpose(1, 2)).transpose(1, 2)
        flows_k = self.max_residue_magnitude * torch.tanh(flows_k)

        HD = self.deformable_groups
        assert HD == 1
        K = self.attn_size
        #flows_k = rearrange(flows_k,'b 1 (hd k tw) h w -> b hd k tw h w',hd=HD,k=K)
        flows_k = rearrange(flows_k,'b 1 (k tw) h w -> b k tw h w',k=K)
        K = flows_k.shape[-2]
        # print("flows_k.shape,flow.shape: ",flows_k.shape,flow.shape)
        # flows_k = flows_k + flow.flip(2).repeat(1,1,flow.size(2)//2,1,1)
        # flows_k = th.zeros_like(flows_k)
        flows_k = flows_k + flow
        # print("flows_k.shape,flow.shape: ",flows_k.shape,flow.shape)

        # exit()
        # flows_k = rearrange(flows_k,'b k tw h w -> b 1 k (h w) tw')
        # flows_k = rearrange(flows_k,'b k tw h w -> b 1 k tw h w')
        flows_k = rearrange(flows_k,'b k tw h w -> b 1 h w k tw')

        # print("flows_k.shape: ",flows_k.shape)
        # exit()
        # dists_k = th.zeros_like(flows_k[...,0,:,:])
        dists_k = th.zeros_like(flows_k[...,0])
        return dists_k,flows_k

    def paired_vids(self, vid0, vid1, flows, wt, skip_self=False):
        return _paired_vids(self.forward, vid0, vid1, flows, wt, skip_self)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.


    Returns:
        Tensor: Warped image or feature map.
    """

    # -- create mesh grid --
    n, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device),
                                    torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    flow = rearrange(flow,'b two h w -> b h w two')
    vgrid = grid + flow

    # -- scale grid to [-1,1] --
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode,
                           padding_mode=padding_mode, align_corners=align_corners)

    return output

