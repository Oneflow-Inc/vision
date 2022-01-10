import oneflow as flow
import oneflow.nn as nn
from functools import partial


class ConvBnAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_layer: nn.Module = nn.ReLU,
        conv: nn.Module = nn.Conv2d,
        norm_layer: nn.Module = nn.BatchNorm2d,
        bias: bool = False,
        inplace: bool = True,
        **kwargs
    ):
        layers = [conv(in_features, out_features, **kwargs, bias=bias)]
        if norm_layer:
            layers.append(norm_layer(out_features))
        if act_layer:
            layers.append(act_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_features = out_features


ConvBn = partial(ConvBnAct, act_layer=None)
ConvAct = partial(ConvBnAct, norm_layer=None, bias=True)
