from benchmark.net import run_net
from flowvision.models.mlp_mixer import *


def run_mlp_mixer(): return run_net(mlp_mixer_b16_224, [16, 3, 224, 224])
