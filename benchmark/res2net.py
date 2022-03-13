from benchmark.net import *
from flowvision.models.res2net import *


def run_res2net(): return run_net(res2net50_26w_4s, [16, 3, 224, 224])
