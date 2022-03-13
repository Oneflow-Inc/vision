from benchmark.net import *
from flowvision.models.rexnet import *
from flowvision.models.rexnet_lite import *


def run_rexnet(): return run_net(rexnetv1_1_0, [16, 3, 224, 224])
def run_rexnet_lite(): return run_net(rexnet_lite_1_0, [16, 3, 224, 224])
