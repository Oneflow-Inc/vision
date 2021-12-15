"""
Modified from https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net.py
"""
import math

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

from .utils import load_state_dict_from_url
from .registry import ModelCreator


model_urls = {
    "res2net50_26w_4s": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/Res2Net/res2net50_26w_4s.zip",
    "res2net50_26w_6s": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/Res2Net/res2net50_26w_6s.zip",
    "res2net50_26w_8s": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/Res2Net/res2net50_26w_8s.zip",
    "res2net50_48w_2s": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/Res2Net/res2net50_48w_2s.zip",
    "res2net50_14w_8s": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/Res2Net/res2net50_14w_8s.zip",
    "res2net101_26w_4s": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/Res2Net/res2net101_26w_4s.zip",
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        baseWidth=26,
        scale=4,
        stype="normal",
    ):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(
                    width, width, kernel_size=3, stride=stride, padding=1, bias=False
                )
            )
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(
            width * scale, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(x)
        out = self.relu(x)

        spx = flow.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == "stage":
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = flow.cat((out, sp), dim=1)

        if self.scale != 1 and self.stype == "normal":
            out = flow.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == "stage":
            out = flow.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample=downsample,
                stype="stage",
                baseWidth=self.baseWidth,
                scale=self.scale,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _create_res2net(arch, pretrained=False, progress=True, **model_kwargs):
    model = Res2Net(**model_kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def res2net50_26w_4s(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Res2Net-50_26w_4s model.

    .. note::
        Res2Net-50_26w_4s model from the `Res2Net: A New Multi-scale Backbone Architecture <https://arxiv.org/pdf/1904.01169.pdf>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> res2net50_26w_4s = flowvision.models.res2net50_26w_4s(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], baseWidth=26, scale=4, **kwargs
    )
    return _create_res2net(
        "res2net50_26w_4s", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def res2net101_26w_4s(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Res2Net-101_26w_4s model.

    .. note::
        Res2Net-101_26w_4s model from the `Res2Net: A New Multi-scale Backbone Architecture <https://arxiv.org/pdf/1904.01169.pdf>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> res2net101_26w_4s = flowvision.models.res2net101_26w_4s(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        block=Bottle2neck, layers=[3, 4, 23, 3], baseWidth=26, scale=4, **kwargs
    )
    return _create_res2net(
        "res2net101_26w_4s", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def res2net50_26w_6s(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Res2Net-50_26w_6s model.

    .. note::
        Res2Net-50_26w_6s model from the `Res2Net: A New Multi-scale Backbone Architecture <https://arxiv.org/pdf/1904.01169.pdf>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> res2net50_26w_6s = flowvision.models.res2net50_26w_6s(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], baseWidth=26, scale=6, **kwargs
    )
    return _create_res2net(
        "res2net50_26w_6s", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def res2net50_26w_8s(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Res2Net-50_26w_8s model.

    .. note::
        Res2Net-50_26w_8s model from the `Res2Net: A New Multi-scale Backbone Architecture <https://arxiv.org/pdf/1904.01169.pdf>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> res2net50_26w_8s = flowvision.models.res2net50_26w_8s(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], baseWidth=26, scale=8, **kwargs
    )
    return _create_res2net(
        "res2net50_26w_8s", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def res2net50_48w_2s(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Res2Net-50_48w_2s model.

    .. note::
        Res2Net-50_48w_2s model from the `Res2Net: A New Multi-scale Backbone Architecture <https://arxiv.org/pdf/1904.01169.pdf>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> res2net50_48w_2s = flowvision.models.res2net50_48w_2s(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], baseWidth=48, scale=2, **kwargs
    )
    return _create_res2net(
        "res2net50_26w_4s", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def res2net50_14w_8s(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Res2Net-50_14w_8s model.

    .. note::
        Res2Net-50_14w_8s model from the `Res2Net: A New Multi-scale Backbone Architecture <https://arxiv.org/pdf/1904.01169.pdf>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> res2net50_14w_8s = flowvision.models.res2net50_14w_8s(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], baseWidth=14, scale=8, **kwargs
    )
    return _create_res2net(
        "res2net50_14w_8s", pretrained=pretrained, progress=progress, **model_kwargs
    )
