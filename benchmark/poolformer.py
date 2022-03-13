from benchmark.net import run_net
from flowvision.models.poolformer import *


def run_poolformer_s12(): return run_net(poolformer_s12, [16, 3, 224, 224])
def run_poolformer_s24(): return run_net(poolformer_s24, [16, 3, 224, 224])
def run_poolformer_s36(): return run_net(poolformer_s36, [16, 3, 224, 224])
def run_poolformer_m36(): return run_net(poolformer_m36, [16, 3, 224, 224])
def run_poolformer_m48(): return run_net(poolformer_m48, [16, 3, 224, 224])
