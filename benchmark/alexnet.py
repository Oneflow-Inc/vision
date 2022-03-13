from benchmark.net import run_net
from flowvision.models.alexnet import *


def run_alexnet(): return run_net(alexnet, [16, 3, 224, 224])
