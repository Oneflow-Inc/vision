from benchmark.net import run_net
from flowvision.models.crossformer import *


def run_crossformer(): return run_net(
    crossformer_tiny_patch4_group7_224, [16, 3, 224, 224])
