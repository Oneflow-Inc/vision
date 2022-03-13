from benchmark.net import *
from flowvision.models.pvt import *


def run_pvt_tiny(): return run_net(pvt_tiny, [16, 3, 224, 224])
def run_pvt_small(): return run_net(pvt_small, [16, 3, 224, 224])
