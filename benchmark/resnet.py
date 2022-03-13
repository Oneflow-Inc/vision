from flowvision.models.resnet import *
from benchmark.net import *


def run_resnet50(): return run_net(resnet50, [16, 3, 224, 224])
def run_resnet50_32x4d(): return run_net(resnext50_32x4d, [16, 3, 224, 224])
def run_resnet50_2(): return run_net(wide_resnet50_2, [16, 3, 224, 224])
