import oneflow as flow
import oneflow.nn as nn
from functools import partial
from flowvision.layers.build import LAYER_REGISTRY


@LAYER_REGISTRY.register()
class ConvBnAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_layer: nn.Module = nn.ReLU,
        conv: nn.Module = nn.Conv2d,
        normalization: nn.Module = nn.BatchNorm2d,
        bias: bool = False,
        **kwargs
    ):
        super().__init__()
        self.add_module("conv", conv(in_features, out_features, **kwargs, bias=bias))
        if normalization:
            self.add_module("bn", normalization(out_features))
        if act_layer:
            self.add_module("act", act_layer())


ConvBn = partial(ConvBnAct, act_layer=None)
ConvAct = partial(ConvBnAct, normalization=None, bias=True)
