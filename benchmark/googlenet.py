from benchmark.net import *
from flowvision.models.googlenet import *


def run_googlenet(): return run_net(googlenet, [16, 3, 224, 224])
