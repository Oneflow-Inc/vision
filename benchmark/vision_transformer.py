from benchmark.net import *
from flowvision.models.vision_transformer import *


def run_vit_tiny_patch(): return run_net(
    vit_tiny_patch16_224, [16, 3, 224, 224])


def run_vit_small_patch(): return run_net(
    vit_small_patch16_224, [16, 3, 224, 224])


def run_vit_base_patch(): return run_net(
    vit_base_patch16_224, [16, 3, 224, 224])
