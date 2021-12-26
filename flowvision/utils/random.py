"""
Modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/random.py
"""
import random
import numpy as np

import oneflow as flow

def random_seed(seed=42, rank=0):
    flow.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)