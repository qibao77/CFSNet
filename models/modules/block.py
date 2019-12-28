from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
####################
# Basic blocks
####################
def default_conv(in_channels, out_channels, kernel_size, padding, bias=False, init_scale=0.1):
    basic_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
    nn.init.kaiming_normal_(basic_conv.weight.data, a=0, mode='fan_in')
    basic_conv.weight.data *= init_scale
    if basic_conv.bias is not None:
        basic_conv.bias.data.zero_()
    return basic_conv


def default_Linear(in_channels, out_channels, bias=False):
    basic_Linear = nn.Linear(in_channels, out_channels, bias=bias)
    # nn.init.xavier_normal_(basic_Linear.weight.data)
    nn.init.kaiming_normal_(basic_Linear.weight.data, a=0, mode='fan_in')
    basic_Linear.weight.data *= 0.1
    if basic_Linear.bias is not None:
        basic_Linear.bias.data.zero_()
    return basic_Linear


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=False):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 1, 0, bias))  # kernal_size is 1
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 1, 0, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
