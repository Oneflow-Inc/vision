from benchmark.net import run_net
from flowvision.models.cswin import *


def run_cswin(): return run_net(cswin_tiny_224, [16, 3, 224, 224])
