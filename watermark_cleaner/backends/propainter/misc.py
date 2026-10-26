"""Inference initializer adapted from pinned ProPainter model/misc.py."""
import torch.nn as nn

def constant_init(module, val, bias=0):
    if getattr(module, "weight", None) is not None:
        nn.init.constant_(module.weight, val)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)
