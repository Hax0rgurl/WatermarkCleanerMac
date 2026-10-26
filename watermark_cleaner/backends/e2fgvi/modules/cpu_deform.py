"""Torchvision CPU implementation of the MMCV operator contract."""
import torch
from torchvision.ops import DeformConv2d, deform_conv2d

class ModulatedDeformConv2d(DeformConv2d):
    def __init__(self, *args, deform_groups=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.deform_groups = deform_groups

def modulated_deform_conv2d(x, offset, mask, weight, bias, stride=1, padding=0, dilation=1, groups=1, deform_groups=1):
    expected_offset = 2 * weight.shape[2] * weight.shape[3] * deform_groups
    if offset.shape[1] != expected_offset:
        raise ValueError('Offset layout does not match deformable groups')
    if x.shape[1] != weight.shape[1] * groups:
        raise ValueError('Convolution group mismatch')
    return deform_conv2d(x, offset, weight, bias, stride, padding, dilation, mask)
