from benchmark.net import *
from flowvision.models.squeezenet import *


def run_squeezenet(): return run_net(squeezenet1_0, [16, 3, 224, 224])
