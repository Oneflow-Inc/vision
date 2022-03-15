from benchmark.net import run_net
from flowvision.models.convnext import *


def run_convnext(): return run_net(convnext_tiny_224, [16, 3, 224, 224])
