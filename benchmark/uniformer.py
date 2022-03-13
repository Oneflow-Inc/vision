from flowvision.models.uniformer import *
from benchmark.net import *


def run_uniformer_base(): return run_net(uniformer_base, [16, 3, 224, 224])


def run_uniformer_base_ls(): return run_net(
    uniformer_base_ls, [16, 3, 224, 224])


def run_uniformer_small(): return run_net(
    uniformer_small, [16, 3, 224, 224])


def run_uniformer_small_plus(): return run_net(
    uniformer_small_plus, [16, 3, 224, 224])
