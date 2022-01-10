"""
Modified from https://github.com/clovaai/rexnet/blob/master/rexnetv1.py
"""
from math import ceil

import oneflow as flow
import oneflow.nn as nn

from .utils import load_state_dict_from_url
from .registry import ModelCreator


model_urls = {
    "rexnetv1_1_0": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/RexNet/rexnetv1_1_0.zip",
    "rexnetv1_1_3": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/RexNet/rexnetv1_1_3.zip",
    "rexnetv1_1_5": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/RexNet/rexnetv1_1_5.zip",
    "rexnetv1_2_0": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/RexNet/rexnetv1_2_0.zip",
    "rexnetv1_3_0": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/RexNet/rexnetv1_3_0.zip",
}


# TODO: Add Memory Efficient SiLU Module
def ConvBNAct(
    out,
    in_channels,
    channels,
    kernel=1,
    stride=1,
    pad=0,
    num_group=1,
    active=True,
    relu6=False,
):
    out.append(
        nn.Conv2d(
            in_channels, channels, kernel, stride, pad, groups=num_group, bias=False
        )
    )
    out.append(nn.BatchNorm2d(channels))
    if active:
        out.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))


def ConvBNSiLU(out, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1):
    out.append(
        nn.Conv2d(
            in_channels, channels, kernel, stride, pad, groups=num_group, bias=False
        )
    )
    out.append(nn.BatchNorm2d(channels))
    out.append(nn.SiLU(inplace=True))


class SE(nn.Module):
    def __init__(self, in_channels, channels, se_ratio=12):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, channels // se_ratio, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels // se_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // se_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class LinearBottleneck(nn.Module):
    def __init__(
        self, in_channels, channels, t, stride, use_se=True, se_ratio=12, **kwargs
    ):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels

        out = []
        # Point-Wise Conv
        if t != 1:
            dw_channels = in_channels * t
            ConvBNSiLU(out, in_channels=in_channels, channels=dw_channels)
        else:
            dw_channels = in_channels

        # Depth-Wise Conv
        ConvBNAct(
            out,
            in_channels=dw_channels,
            channels=dw_channels,
            kernel=3,
            stride=stride,
            pad=1,
            num_group=dw_channels,
            active=False,
        )

        # SE Module
        if use_se:
            out.append(SE(dw_channels, dw_channels, se_ratio))

        out.append(nn.ReLU6())

        # Point-Wise Conv without Activation
        ConvBNAct(
            out, in_channels=dw_channels, channels=channels, active=False, relu6=True
        )
        self.out = nn.Sequential(*out)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out[:, 0 : self.in_channels] += x

        return out


class RexNetV1(nn.Module):
    def __init__(
        self,
        input_ch=16,
        final_ch=180,
        width_mult=1.0,
        depth_mult=1.0,
        classes=1000,
        use_se=True,
        se_ratio=12,
        dropout_ratio=0.2,
        bn_momentum=0.9,
    ):
        super(RexNetV1, self).__init__()

        layers = [
            1,
            2,
            2,
            3,
            3,
            5,
        ]  # stage-depth, e.g., the first stage has only one block, and the second stage has two blocks.
        strides = [1, 2, 2, 2, 1, 2]  # the strides of the first block of each stage
        use_ses = [False, False, True, True, True, True]

        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum(
            [
                [element] + [1] * (layers[idx] - 1)
                for idx, element in enumerate(strides)
            ],
            [],
        )
        if use_se:
            use_ses = sum(
                [[element] * layers[idx] for idx, element in enumerate(use_ses)], []
            )
        else:
            use_ses = [False] * sum(layers[:])
        ts = [1] * layers[0] + [6] * sum(layers[1:])

        self.depth = sum(layers[:]) * 3
        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = input_ch / width_mult if width_mult < 1.0 else input_ch

        features = []
        in_channels_group = []
        channels_group = []

        # The following channel configuration is a simple instance to make each layer become an expand layer.
        for i in range(self.depth // 3):
            if i == 0:
                in_channels_group.append(int(round(stem_channel * width_mult)))
                channels_group.append(int(round(inplanes * width_mult)))
            else:
                in_channels_group.append(int(round(inplanes * width_mult)))
                inplanes += final_ch / (self.depth // 3 * 1.0)
                channels_group.append(int(round(inplanes * width_mult)))

        ConvBNSiLU(
            features,
            3,
            int(round(stem_channel * width_mult)),
            kernel=3,
            stride=2,
            pad=1,
        )

        for block_idx, (in_c, c, t, s, se) in enumerate(
            zip(in_channels_group, channels_group, ts, strides, use_ses)
        ):
            features.append(
                LinearBottleneck(
                    in_channels=in_c,
                    channels=c,
                    t=t,
                    stride=s,
                    use_se=se,
                    se_ratio=se_ratio,
                )
            )
        pen_channels = int(1280 * width_mult)
        ConvBNSiLU(features, c, pen_channels)

        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)
        self.output = nn.Sequential(
            nn.Dropout(dropout_ratio), nn.Conv2d(pen_channels, classes, 1, bias=True)
        )

    def extract_features(self, x):
        return self.features[:-1](x)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x).flatten(1)
        return x


def _create_rexnetv1(arch, pretrained=False, progress=True, **model_kwargs):
    model = RexNetV1(**model_kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def rexnetv1_1_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ReXNet model with width multiplier of 1.0.

    .. note::
        ReXNet model with width multiplier of 1.0 from the `Rethinking Channel Dimensions for Efficient Model Design <https://arxiv.org/pdf/2007.00992.pdf>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> rexnetv1_1_0 = flowvision.models.rexnetv1_1_0(pretrained=False, progress=True)

    """
    model_kwargs = dict(width_mult=1.0, **kwargs)
    return _create_rexnetv1(
        "rexnetv1_1_0", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def rexnetv1_1_3(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ReXNet model with width multiplier of 1.3.

    .. note::
        ReXNet model with width multiplier of 1.3 from the `Rethinking Channel Dimensions for Efficient Model Design <https://arxiv.org/pdf/2007.00992.pdf>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> rexnetv1_1_3 = flowvision.models.rexnetv1_1_3(pretrained=False, progress=True)

    """
    model_kwargs = dict(width_mult=1.3, **kwargs)
    return _create_rexnetv1(
        "rexnetv1_1_3", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def rexnetv1_1_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ReXNet model with width multiplier of 1.5.

    .. note::
        ReXNet model with width multiplier of 1.5 from the `Rethinking Channel Dimensions for Efficient Model Design <https://arxiv.org/pdf/2007.00992.pdf>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> rexnetv1_1_5 = flowvision.models.rexnetv1_1_5(pretrained=False, progress=True)

    """
    model_kwargs = dict(width_mult=1.5, **kwargs)
    return _create_rexnetv1(
        "rexnetv1_1_5", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def rexnetv1_2_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ReXNet model with width multiplier of 2.0.

    .. note::
        ReXNet model with width multiplier of 2.0 from the `Rethinking Channel Dimensions for Efficient Model Design <https://arxiv.org/pdf/2007.00992.pdf>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> rexnetv1_2_0 = flowvision.models.rexnetv1_2_0(pretrained=False, progress=True)

    """
    model_kwargs = dict(width_mult=2.0, **kwargs)
    return _create_rexnetv1(
        "rexnetv1_2_0", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def rexnetv1_3_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ReXNet model with width multiplier of 3.0.

    .. note::
        ReXNet model with width multiplier of 3.0 from the `Rethinking Channel Dimensions for Efficient Model Design <https://arxiv.org/pdf/2007.00992.pdf>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> rexnetv1_3_0 = flowvision.models.rexnetv1_3_0(pretrained=False, progress=True)

    """
    model_kwargs = dict(width_mult=3.0, **kwargs)
    return _create_rexnetv1(
        "rexnetv1_3_0", pretrained=pretrained, progress=progress, **model_kwargs
    )
