from flowvision.models.res_mlp import *
from benchmark.net import *


def run_res_mlp(): return run_net(resmlp_12_224, [16, 3, 224, 224])
