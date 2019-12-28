from torch import nn
import torch
import math
from .block import *

class TuningBlock(nn.Module):
    def __init__(self, input_size):
        super(TuningBlock, self).__init__()
        self.conv0 = default_conv(in_channels=input_size, out_channels=input_size,
                                  kernel_size=3, padding=1, bias=False, init_scale=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = default_conv(in_channels=input_size, out_channels=input_size,
                                  kernel_size=3, padding=1, bias=False, init_scale=0.1)

    def forward(self, x):
        out = self.conv0(x)
        out = self.relu0(out)
        out = self.conv1(out)
        return out


class TuningBlockModule(nn.Module):
    def __init__(self, channels=64, num_blocks=5, task_type='sr', upscale=4):
        super(TuningBlockModule, self).__init__()
        self.num_channels = channels
        self.task_type = task_type
        # define control variable
        self.control_alpha = nn.Sequential(
            default_Linear(512, 256, bias=False),
            default_Linear(256, 128, bias=False),
            default_Linear(128, channels, bias=False)
        )
        self.adaptive_alpha = nn.ModuleList(
            [nn.Sequential(
                default_Linear(channels, channels, bias=False),
                default_Linear(channels, channels, bias=False)
            ) for _ in range(num_blocks)]
        )
        self.tuning_blocks = nn.ModuleList(
            [TuningBlock(channels) for _ in range(num_blocks)]
        )
        if self.task_type == 'sr':
            self.tuning_blocks.append(nn.Sequential(
                default_conv(in_channels=channels, out_channels=channels,
                             kernel_size=3, padding=1, bias=False, init_scale=0.1),
                Upsampler(default_conv, upscale, channels, bias=False, act=False),
            ))
            self.adaptive_alpha.append(nn.Sequential(
                default_Linear(channels, channels, bias=False),
                default_Linear(channels, channels, bias=False)
            ))

    def forward(self, x, alpha, number=0):
        input_alpha = self.control_alpha(alpha)
        tuning_f = self.tuning_blocks[number](x)
        ad_alpha = self.adaptive_alpha[number](input_alpha)
        ad_alpha = ad_alpha.view(-1, self.num_channels, 1, 1)
        return tuning_f * ad_alpha, torch.ones_like(ad_alpha).cuda()-ad_alpha
