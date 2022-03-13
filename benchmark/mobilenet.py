from benchmark.net import *
from flowvision.models.mobilenet_v3  import *
from flowvision.models.mobilenet_v2  import *

def run_mobilenet_v3(): return run_net(mobilenet_v3_large, [16,3,224,224]) 
def run_mobilenet_v2(): return run_net(mobilenet_v2, [16,3,224,224]) 
