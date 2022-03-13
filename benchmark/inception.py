from benchmark.net import *
from flowvision.models.inception_v3 import *


def run_inception_v3(): return run_net(inception_v3, [16, 3, 299, 299])
