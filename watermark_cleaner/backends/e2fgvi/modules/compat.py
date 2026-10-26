"""Small inference-only equivalents of MMCV's used convolution helpers.

State names and computation match the five SPyNet ConvModules exactly:
Conv2d with bias, optional inplace ReLU, no normalization layer.
"""
import torch.nn as nn

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, norm_cfg=None, act_cfg=None):
        super().__init__()
        if norm_cfg is not None:
            raise ValueError("SPyNet inference expects no normalization")
        if act_cfg not in (None, {"type": "ReLU"}):
            raise ValueError("Unsupported SPyNet activation")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.activate = nn.ReLU(inplace=True) if act_cfg else nn.Identity()

    def forward(self, tensor):
        return self.activate(self.conv(tensor))


def constant_init(module, val, bias=0):
    if getattr(module, "weight", None) is not None:
        nn.init.constant_(module.weight, val)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)
