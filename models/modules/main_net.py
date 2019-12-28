from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .block import *

####################
# Useful blocks
####################
class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size, padding=1,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, padding=padding, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res = res + x

        return res

class MainNet(nn.Module):
    def __init__(self, n_colors, out_nc, num_channels, num_blocks, task_type='sr', upscale=4):
        super(MainNet, self).__init__()

        self.task_type = task_type
        # define head
        self.head = default_conv(in_channels=n_colors, out_channels=num_channels,
                                 kernel_size=3, padding=1, bias=False, init_scale=0.1)
        self.body = nn.ModuleList(
            [ResBlock(default_conv,
                      n_feats=num_channels, kernel_size=3, act=nn.ReLU(True), res_scale=1
                      ) for _ in range(num_blocks)]
        )

        if self.task_type == 'sr':
            self.tail = nn.Sequential(
                default_conv(in_channels=num_channels, out_channels=num_channels,
                             kernel_size=3, padding=1, bias=False, init_scale=0.1),
                Upsampler(default_conv, upscale, num_channels, act=False),
            )

        self.end = default_conv(in_channels=num_channels, out_channels=out_nc,
                                kernel_size=3, padding=1, bias=False, init_scale=0.1)

    def forward(self, x):
        output = self.head(x)
        head_f = output
        for mbody in self.body:
            output = mbody(output)
        if self.task_type == 'sr':
            output = self.tail(output)
        output = self.end(output + head_f)
        return output