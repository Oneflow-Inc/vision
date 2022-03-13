from benchmark.net import *
from flowvision.models.swin_transformer import *


def run_swin_tiny_patch(): return run_net(
    swin_tiny_patch4_window7_224, [16, 3, 224, 224])


def run_swin_small_patch(): return run_net(
    swin_small_patch4_window7_224, [16, 3, 224, 224])


def run_swin_base_patch(): return run_net(
    swin_base_patch4_window7_224, [16, 3, 224, 224])
