"""
Modified from https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pyflow/g_ghost_regnet.py
"""
from typing import Type, Any, Callable, Union, List, Optional
import numpy as np

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
from oneflow import Tensor

from .registry import ModelCreator
from .utils import load_state_dict_from_url


__all__ = ["g_ghost_regnet"]


model_urls = {
    "g_ghost_regnet": None,
}


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, 
        out_planes, 
        kernel_size=3, 
        stride=stride,
        padding=dilation, 
        groups=groups, 
        bias=False, 
        dilation=dilation
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion: int = 1

    def __init__(
        self, 
        inplanes: int, 
        planes: int, 
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        group_width: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = planes * self.expansion
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, width // min(width, group_width), dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class GGhostStage(nn.Module):
    def __init__(
        self, 
        block: Type[Union[Bottleneck]], 
        inplanes, 
        planes, 
        group_width, 
        blocks, 
        stride=1, 
        dilate=False,
        norm_layer=None,
        downsample=None,
        cheap_ratio=0.5
    ) -> None:
        super(GGhostStage, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.dilation = 1
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )

        self.conv_first = block(
            inplanes, 
            planes, 
            stride, 
            downsample, 
            group_width,
            previous_dilation, 
            norm_layer
        )
        self.conv_last = block(
            planes, 
            planes, 
            group_width=group_width,
            dilation=self.dilation,
            norm_layer=norm_layer
        )

        group_width = int(group_width * 0.75)
        raw_planes = int(planes * (1 - cheap_ratio) / group_width) * group_width
        cheap_planes = planes - raw_planes
        self.cheap_planes = cheap_planes
        self.raw_planes = raw_planes
        
        self.merge = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(planes+raw_planes*(blocks-2), cheap_planes),
            norm_layer(cheap_planes),
            nn.ReLU(inplace=True),
            conv1x1(cheap_planes, cheap_planes),
            norm_layer(cheap_planes),
            # nn.ReLU(inplace=True),
        )
        self.cheap = nn.Sequential(
            conv1x1(cheap_planes, cheap_planes),
            norm_layer(cheap_planes),
            # nn.ReLU(inplace=True),
        )
        self.cheap_relu = nn.ReLU(inplace=True)
        
        layers = []
        downsample = nn.Sequential(
            LambdaLayer(lambda x: x[:, :raw_planes])
        )

        layers = []
        layers.append(
            block(
                raw_planes, 
                raw_planes, 
                1, 
                downsample, 
                group_width,
                self.dilation, 
                norm_layer
            )
        )
        inplanes = raw_planes
        for _ in range(2, blocks-1):
            layers.append(
                block(
                    inplanes, 
                    raw_planes, 
                    group_width=group_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        self.layers = nn.Sequential(*layers)
    
    def forward(self, input):
        x0 = self.conv_first(input)

        m_list = [x0]
        e = x0[:, :self.raw_planes]        
        for l in self.layers:
            e = l(e)
            m_list.append(e)
        m = flow.cat(m_list, dim=1)
        m = self.merge(m)
        
        c = x0[:, self.raw_planes:]
        c = self.cheap_relu(self.cheap(c) + m)

        x = flow.cat([e, c], dim=1)
        x = self.conv_last(x)
        return x


class GhostRegNet(nn.Module):
    def __init__(
        self, 
        block, 
        layers, 
        widths, 
        num_classes=1000, 
        zero_init_residual=True,
        group_width=1, 
        replace_stride_with_dilation=None,
        norm_layer=None
    ) -> None:
        super(GhostRegNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.group_width = group_width
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(
            block, widths[0], layers[0], stride=2, 
            dilate=replace_stride_with_dilation[0]
        )

        self.inplanes = widths[0]
        if layers[1] > 2:
            self.layer2 = GGhostStage(
                block, 
                self.inplanes, 
                widths[1], 
                group_width, 
                layers[1], 
                stride=2,
                dilate=replace_stride_with_dilation[1], 
                cheap_ratio=0.5
            )
        else:      
            self.layer2 = self._make_layer(
                block, widths[1], layers[1], stride=2,
                dilate=replace_stride_with_dilation[1]
            )
        
        self.inplanes = widths[1]
        self.layer3 = GGhostStage(
            block, 
            self.inplanes, 
            widths[2], 
            group_width, 
            layers[2], 
            stride=2,
            dilate=replace_stride_with_dilation[2], 
            cheap_ratio=0.5
        )
        
        self.inplanes = widths[2]
        if layers[3] > 2:
            self.layer4 = GGhostStage(
                block, 
                self.inplanes, 
                widths[3], 
                group_width, 
                layers[3], 
                stride=2,
                dilate=replace_stride_with_dilation[3], 
                cheap_ratio=0.5
            ) 
        else:
            self.layer4 = self._make_layer(
                block, widths[3], layers[3], stride=2, 
                dilate=replace_stride_with_dilation[3]
            )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(widths[-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # see: https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/g_ghost_regnet.py#L224
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self, 
        block: Type[Union[Bottleneck]], 
        planes: int, 
        blocks: int, 
        stride:int = 1, 
        dilate: bool = False
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, 
                planes, 
                stride, 
                downsample, 
                self.group_width,
                previous_dilation, 
                norm_layer
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, 
                    planes, 
                    group_width=self.group_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _g_ghost_regnet(
    block: Type[Union[Bottleneck]],
    layers: List[int],
    widths: List[int],
    group_width: int = 48,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> GhostRegNet:
    model = GhostRegNet(block, layers, widths, group_width=group_width, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def g_ghost_regnet(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    """
    Constructs the G_GhostNet model.

    .. note::
        G_GhostNet model from `GhostNets on Heterogeneous Devices via Cheap Operations <https://arxiv.org/abs/2201.03297v1>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> ghostnet = flowvision.models.g_ghost_regnet(pretrained=True, progress=True)

    """
    layers = [2, 6, 15, 2]
    widths = [96, 192, 432, 1008]
    return _g_ghost_regnet(Bottleneck, layers, widths, **kwargs)
