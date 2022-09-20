# This file copyed from https://github.com/pytorch/vision/blob/main/hubconf.py
# Optional list of dependencies required by the package
dependencies = ["oneflow"]

from flowvision.models.alexnet import alexnet
from flowvision.models.densenet import densenet121, densenet161, densenet169, densenet201
from flowvision.models.efficientnet import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7
)
from flowvision.models.googlenet import googlenet
from flowvision.models.inception_v3 import inception_v3
from flowvision.models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
from flowvision.models.mobilenet_v2 import mobilenet_v2
from flowvision.models.mobilenet_v3 import mobilenet_v3_large, mobilenet_v3_small
from flowvision.models.regnet import (
    regnet_x_16gf,
    regnet_x_1_6gf,
    regnet_x_32gf,
    regnet_x_3_2gf,
    regnet_x_400mf,
    regnet_x_800mf,
    regnet_x_8gf,
    regnet_y_128gf,
    regnet_y_16gf,
    regnet_y_1_6gf,
    regnet_y_32gf,
    regnet_y_3_2gf,
    regnet_y_400mf,
    regnet_y_800mf,
    regnet_y_8gf,
)
from flowvision.models.resnet import (
    resnet101,
    resnet152,
    resnet18,
    resnet34,
    resnet50,
    resnext101_32x8d,
    resnext50_32x4d,
    wide_resnet101_2,
    wide_resnet50_2,
)
from flowvision.models.shufflenet_v2 import (
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)
from flowvision.models.squeezenet import squeezenet1_0, squeezenet1_1
from flowvision.models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
