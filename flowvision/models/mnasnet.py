"""
Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py
"""
import math
import warnings

import oneflow as flow
import oneflow.nn as nn

from .utils import load_state_dict_from_url
from .registry import ModelCreator

__all__ = ["MNASNet", "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"]

model_urls = {
    "mnasnet0_5": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/MNASNet/mnasnet0_5.zip",
    "mnasnet0_75": None,
    "mnasnet1_0": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/MNASNet/mnasnet1_0.zip",
    "mnasnet1_3": None,
}

_BN_MOMENTUM = 1 - 0.9997


class _InvertedResidual(nn.Module):
    def __init__(
        self, in_ch, out_ch, kernel_size, stride, expansion_factor, bn_momentum=0.1
    ):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_resudual = in_ch == out_ch and stride == 1
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(
                mid_ch,
                mid_ch,
                kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                groups=mid_ch,
                bias=False,
            ),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Linear pointwise, Note that there's no activation
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum),
        )

    def forward(self, input):
        if self.apply_resudual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats, bn_momentum):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(
        in_ch, out_ch, kernel_size, stride, exp_factor, bn_momentum=bn_momentum
    )
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            _InvertedResidual(
                out_ch, out_ch, kernel_size, 1, exp_factor, bn_momentum=bn_momentum
            )
        )
    return nn.Sequential(first, *remaining)


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MNASNet(nn.Module):
    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2

    def __init__(self, alpha, num_classes=1000, dropout=0.2):
        super(MNASNet, self).__init__()
        assert alpha > 0.0
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths(alpha)
        layers = [
            # First layer: regular conv.
            nn.Conv2d(3, depths[0], 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # Depthwise separable, no skip.
            nn.Conv2d(
                depths[0],
                depths[0],
                3,
                padding=1,
                stride=1,
                groups=depths[0],
                bias=False,
            ),
            nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(depths[0], depths[1], 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(depths[1], momentum=_BN_MOMENTUM),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[2], depths[3], 5, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM),
            _stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM),
            _stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM),
            _stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM),
            # Final mapping to classifier input.
            nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        ]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True), nn.Linear(1280, num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_out", nonlinearity="sigmoid"
                )
                nn.init.zeros_(m.bias)


def _load_pretrained(model_name, model, progress):
    if model_name not in model_urls or model_urls[model_name] is None:
        raise ValueError(
            "No checkpoint is available for model type {}".format(model_name)
        )
    checkpoint_url = model_urls[model_name]
    model.load_state_dict(load_state_dict_from_url(checkpoint_url, progress=progress))


@ModelCreator.register_model
def mnasnet0_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs the MNASNet model with depth multiplier of 0.5.

    .. note::
        MNASNet model with depth multiplier of 0.5 from the `MnasNet: Platform-Aware Neural Architecture Search for Mobile <https://arxiv.org/pdf/1807.11626.pdf>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mnasnet0_5 = flowvision.models.mnasnet0_5(pretrained=False, progress=True)

    """
    model = MNASNet(0.5, **kwargs)
    if pretrained:
        _load_pretrained("mnasnet0_5", model, progress)
    return model


@ModelCreator.register_model
def mnasnet0_75(pretrained=False, progress=True, **kwargs):
    """
    Constructs the MNASNet model with depth multiplier of 0.75.

    .. note::
        MNASNet model with depth multiplier of 0.75 from the `MnasNet: Platform-Aware Neural Architecture Search for Mobile <https://arxiv.org/pdf/1807.11626.pdf>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mnasnet0_75 = flowvision.models.mnasnet0_75(pretrained=False, progress=True)

    """
    model = MNASNet(0.75, **kwargs)
    if pretrained:
        _load_pretrained("mnasnet0_75", model, progress)
    return model


@ModelCreator.register_model
def mnasnet1_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs the MNASNet model with depth multiplier of 1.0.

    .. note::
        MNASNet model with depth multiplier of 1.0 from the `MnasNet: Platform-Aware Neural Architecture Search for Mobile <https://arxiv.org/pdf/1807.11626.pdf>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mnasnet1_0 = flowvision.models.mnasnet1_0(pretrained=False, progress=True)

    """
    model = MNASNet(1.0, **kwargs)
    if pretrained:
        _load_pretrained("mnasnet1_0", model, progress)
    return model


@ModelCreator.register_model
def mnasnet1_3(pretrained=False, progress=True, **kwargs):
    """
    Constructs the MNASNet model with depth multiplier of 1.3.

    .. note::
        MNASNet model with depth multiplier of 1.3 from the `MnasNet: Platform-Aware Neural Architecture Search for Mobile <https://arxiv.org/pdf/1807.11626.pdf>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mnasnet1_3 = flowvision.models.mnasnet1_3(pretrained=False, progress=True)

    """
    model = MNASNet(1.3, **kwargs)
    if pretrained:
        _load_pretrained("mnasnet1_3", model, progress)
    return model
