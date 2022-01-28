"""
Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
"""
from typing import Union, List, Dict, Any, cast

import oneflow as flow
import oneflow.nn as nn

from .utils import load_state_dict_from_url
from .registry import ModelCreator


__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]


model_urls = {
    "vgg11": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VGG/vgg11.zip",
    "vgg13": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VGG/vgg13.zip",
    "vgg16": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VGG/vgg16.zip",
    "vgg19": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VGG/vgg19.zip",
    "vgg11_bn": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VGG/vgg11_bn.zip",
    "vgg13_bn": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VGG/vgg13_bn.zip",
    "vgg16_bn": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VGG/vgg16_bn.zip",
    "vgg19_bn": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VGG/vgg19_bn.zip",
}


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(
    arch: str,
    cfg: str,
    batch_norm: bool,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    """
    Constructs the VGG-11 model (configuration "A").

    .. note::
        VGG 11-layer model (configuration “A”) from `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
        The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> vgg11 = flowvision.models.vgg11(pretrained=False, progress=True)

    """
    return _vgg("vgg11", "A", False, pretrained, progress, **kwargs)


@ModelCreator.register_model
def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    """
    Constructs the VGG-11 model with batch normalization (configuration "A").

    .. note::
        VGG 11-layer model (configuration “A”) with batch normalization `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
        The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> vgg11_bn = flowvision.models.vgg11_bn(pretrained=False, progress=True)

    """
    return _vgg("vgg11_bn", "A", True, pretrained, progress, **kwargs)


@ModelCreator.register_model
def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    """
    Constructs the VGG-13 model (configuration "B").

    .. note::
        VGG 13-layer model (configuration “B”) from `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
        The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> vgg13 = flowvision.models.vgg13(pretrained=False, progress=True)

    """
    return _vgg("vgg13", "B", False, pretrained, progress, **kwargs)


@ModelCreator.register_model
def vgg13_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    """
    Constructs the VGG-13 model with batch normalization (configuration "B").

    .. note::
        VGG 13-layer model (configuration “B”) with batch normalization from `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
        The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> vgg13_bn = flowvision.models.vgg13_bn(pretrained=False, progress=True)

    """
    return _vgg("vgg13_bn", "B", True, pretrained, progress, **kwargs)


@ModelCreator.register_model
def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    """
    Constructs the VGG-16 model (configuration "D").

    .. note::
        VGG 16-layer model (configuration “D”) from `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
        The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> vgg16 = flowvision.models.vgg16(pretrained=False, progress=True)

    """
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)


@ModelCreator.register_model
def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    """
    Constructs the VGG-16 model (configuration "D") with batch normalization.

    .. note::
        VGG 16-layer model (configuration “D”) with batch normalization from `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
        The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> vgg16_bn = flowvision.models.vgg16_bn(pretrained=False, progress=True)

    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, **kwargs)


@ModelCreator.register_model
def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    """
    Constructs the VGG-19 model (configuration "E").

    .. note::
        VGG 19-layer model (configuration “E”) from `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
        The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> vgg19 = flowvision.models.vgg19(pretrained=False, progress=True)

    """
    return _vgg("vgg19", "E", False, pretrained, progress, **kwargs)


@ModelCreator.register_model
def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    """
    Constructs the VGG-19 model (configuration "E") with batch normalization.

    .. note::
        VGG 19-layer model (configuration “E”) with batch normalization from `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
        The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> vgg19_bn = flowvision.models.vgg19_bn(pretrained=False, progress=True)

    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, **kwargs)
