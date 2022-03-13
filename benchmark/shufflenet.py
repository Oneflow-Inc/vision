from benchmark.net import *
from flowvision.models.shufflenet_v2 import *


def run_shufflenet(): return run_net(shufflenet_v2_x0_5, [16, 3, 224, 224])
