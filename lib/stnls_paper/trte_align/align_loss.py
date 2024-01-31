

import torch
import torch as th
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class AlignLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,vid,flow):
        print("vid.shape: ",vid.shape)
        print("flow.shape: ",flow.shape)
        exit()
        pass

