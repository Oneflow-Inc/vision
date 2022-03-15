from benchmark.net import *
from flowvision.models.densenet import *


def run_densenet(): return run_net(densenet121, [16, 3, 224, 224])
