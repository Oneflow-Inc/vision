"""
Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenet.py
"""
from .mobilenet_v2 import MobileNetV2, mobilenet_v2, __all__ as mv2_all
from .mobilenet_v3 import (
    MobileNetV3,
    mobilenet_v3_large,
    mobilenet_v3_small,
    __all__ as mv3_all,
)

__all__ = mv2_all + mv3_all
