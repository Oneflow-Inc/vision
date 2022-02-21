"""
Modified from https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/models/resnest.py
"""
import math

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

from .utils import load_state_dict_from_url
from .registry import ModelCreator


__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']

model_urls = {
    "resnest50" : "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNeSt/ResNeSt_50.zip",
    "resnest101": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNeSt/ResNeSt_101.zip",
    "resnest200": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNeSt/ResNeSt_200.zip",
    "resnest269": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNeSt/ResNeSt_269.zip",
}


class SplAtConv2d(nn.Module):
    r""" Split-Attention Conv2d. 
    
    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of channels in each split whin a cardinal group.
        kernel_size (int): Size of convolutional kernel. 
        stride (tuple(int)): Stride of convolution. Default: (1, 1)
        padding (tuple(int)): Padding of convolution. Default: (0, 0)
        dilation (tuple(int)): Dilation of convolution. Default: (1, 1)
        groups (int): Number of featuremap cardinal groups. Default: 1
        bias (bool): Default: True
        radix (int): Number of splits within a cardinal group. Default: 2
        reduction_factor (int): Reduction factor. Default: 4
        norm_layer: Normalization layer used in backbone network. Default: nn.BatchNorm2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 norm_layer=None, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = nn.modules.utils._pair(padding)
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        
        self.cardinality = groups
        self.channels = channels
        self.conv = nn.Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation, groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = flow.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited) 
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = flow.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = flow.sigmoid(x)
        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class ResNestBottleneck(nn.Module):
    """ResNest Bottleneck
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 norm_layer=None, last_gamma=False):
        super(ResNestBottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, norm_layer=norm_layer,
                )
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            nn.init.zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNest(nn.Module):
    """ResNest: 
    The OneFlow impl of : 'ResNeSt: Split-Attention Networks' - 
        https://arxiv.org/pdf/2004.08955.pdf
        
    Args:
        block: Class for the residual block. Default: ResNestBottleneck
        layers (list(int)) : Numbers of layers in each block.
        radix (int): Number of splits within a cardinal group. Default: 2
        groups (int): Number of featuremap cardinal groups. Default: 1
    """
    def __init__(self, block, layers, radix=2, groups=1, bottleneck_width=64,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 avd=False, avd_first=False,final_drop=0.0, 
                 last_gamma=False, norm_layer=nn.BatchNorm2d):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(ResNest, self).__init__()
        
        conv_layer = nn.Conv2d

        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer)
        elif dilation==2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zeros_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, norm_layer=norm_layer,
                                last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first, norm_layer=norm_layer, 
                                last_gamma=self.last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, norm_layer=norm_layer,
                                last_gamma=self.last_gamma))

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
        x = flow.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)
        return x


def _create_resnest(arch, pretrained=False, progress=True, **model_kwargs):
    model = ResNest(**model_kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def resnest50(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ResNest-50 model trained on ImageNet2012.

    .. note::
        ResNest-50 model from `"ResNeSt: Split-Attention Networks" <https://arxiv.org/abs/2004.08955>` _.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> resnest50 = flowvision.models.resnest50(pretrained=False, progress=True)

    """
    model_kwargs = dict(block=ResNestBottleneck, layers=[3, 4, 6, 3], 
                        radix=2, groups=1, bottleneck_width=64, 
                        deep_stem=True, stem_width=32, avg_down=True, 
                        avd=True, avd_first=False, **kwargs)
    return _create_resnest(
        "resnest50", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def resnest101(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ResNest-101 model trained on ImageNet2012.

    .. note::
        ResNest-101 model from `"ResNeSt: Split-Attention Networks" <https://arxiv.org/abs/2004.08955>` _.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> resnest101 = flowvision.models.resnest101(pretrained=False, progress=True)

    """
    model_kwargs = dict(block=ResNestBottleneck, layers=[3, 4, 23, 3], 
                        radix=2, groups=1, bottleneck_width=64, 
                        deep_stem=True, stem_width=64, avg_down=True, 
                        avd=True, avd_first=False, **kwargs)
    return _create_resnest(
        "resnest101", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def resnest200(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ResNest-200 model trained on ImageNet2012.

    .. note::
        ResNest-200 model from `"ResNeSt: Split-Attention Networks" <https://arxiv.org/abs/2004.08955>` _.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> resnest200 = flowvision.models.resnest200(pretrained=False, progress=True)

    """
    model_kwargs = dict(block=ResNestBottleneck, layers=[3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    return _create_resnest(
        "resnest200", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def resnest269(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ResNest-269 model trained on ImageNet2012.

    .. note::
        ResNest-269 model from `"ResNeSt: Split-Attention Networks" <https://arxiv.org/abs/2004.08955>` _.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> resnest269 = flowvision.models.resnest269(pretrained=False, progress=True)

    """
    model_kwargs = dict(block=ResNestBottleneck, layers=[3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    return _create_resnest(
        "resnest269", pretrained=pretrained, progress=progress, **model_kwargs
    )
