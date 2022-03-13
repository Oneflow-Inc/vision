from benchmark.net import *
from flowvision.models.mnasnet import *


def run_mnasnet(): return run_net(mnasnet0_5, [16, 3, 224, 224])
