from benchmark.net import *
from flowvision.models.ghostnet import *


def run_ghostnet(): return run_net(ghostnet, [16, 3, 224, 224])
