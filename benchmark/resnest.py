from flowvision.models.resnest import *
from benchmark.net import *


def run_resnest50(): return run_net(resnest50, [16, 3, 224, 224])
def run_resnest101(): return run_net(resnest101, [16, 3, 256, 256])
def run_resnest200(): return run_net(resnest200, [16, 3, 320, 320])
def run_resnest269(): return run_net(resnest269, [16, 3, 416, 416])
