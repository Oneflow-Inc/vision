"""
Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
"""
import copy
import math
from functools import partial
from typing import Any, Callable, Optional, List, Sequence

import oneflow as flow
import oneflow.nn as nn
from oneflow import Tensor

from .registry import ModelCreator
from .helpers import make_divisible
from .utils import load_state_dict_from_url
from flowvision.layers.regularization import StochasticDepth
from flowvision.layers.blocks import ConvBnAct


__all__ = [
    "EfficientNet",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
]


model_urls = {
    "efficientnet_b0": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/EfficientNet/efficientnet_b0.zip",
    "efficientnet_b1": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/EfficientNet/efficientnet_b1.zip",
    "efficientnet_b2": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/EfficientNet/efficientnet_b2.zip",
    "efficientnet_b3": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/EfficientNet/efficientnet_b3.zip",
    "efficientnet_b4": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/EfficientNet/efficientnet_b4.zip",
    "efficientnet_b5": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/EfficientNet/efficientnet_b5.zip",
    "efficientnet_b6": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/EfficientNet/efficientnet_b6.zip",
    "efficientnet_b7": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/EfficientNet/efficientnet_b7.zip",
}


class SqueezeExcitation(nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in in eq. 3.
    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., flow.nn.Module], optional): ``delta`` activation. Default: ``flow.nn.ReLU``
        scale_activation (Callable[..., flow.nn.Module]): ``sigma`` activation. Default: ``flow.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float,
        depth_mult: float,
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "expand_ratio={expand_ratio}"
        s += ", kernel={kernel}"
        s += ", stride={stride}"
        s += ", input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)

    @staticmethod
    def adjust_channels(
        channels: int, width_mult: float, min_value: Optional[int] = None
    ) -> int:
        return make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )
        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvBnAct(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    act_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvBnAct(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                act_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(
            se_layer(
                expanded_channels,
                squeeze_channels,
                activation=partial(nn.SiLU, inplace=True),
            )
        )

        # project
        layers.append(
            ConvBnAct(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                act_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[MBConvConfig],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet main class
        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[MBConvConfig]"
            )

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvBnAct(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                act_layer=nn.SiLU,
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                )

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 4 * lastconv_input_channels
        layers.append(
            ConvBnAct(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                act_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = flow.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _efficientnet(
    arch: str,
    width_mult: float,
    depth_mult: float,
    dropout: float,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> EfficientNet:
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    model = EfficientNet(inverted_residual_setting, dropout, **kwargs)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def efficientnet_b0(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """
    Constructs the EfficientNet B0 model.

    .. note::
        EfficientNet B0 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
        Note that the (resize-size, crop-size) should be (256, 224) for efficientnet-b0 model when training and testing.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> efficientnet_b0 = flowvision.models.efficientnet_b0(pretrained=False, progress=True)

    """
    return _efficientnet(
        "efficientnet_b0", 1.0, 1.0, 0.2, pretrained, progress, **kwargs
    )


@ModelCreator.register_model
def efficientnet_b1(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """
    Constructs the EfficientNet B1 model.

    .. note::
        EfficientNet B1 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
        Note that the (resize-size, crop-size) should be (256, 240) for efficientnet-b1 model when training and testing.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> efficientnet_b1 = flowvision.models.efficientnet_b1(pretrained=False, progress=True)

    """
    return _efficientnet(
        "efficientnet_b1", 1.0, 1.1, 0.2, pretrained, progress, **kwargs
    )


@ModelCreator.register_model
def efficientnet_b2(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """
    Constructs the EfficientNet B2 model.

    .. note::
        EfficientNet B2 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
        Note that the (resize-size, crop-size) should be (288, 288) for efficientnet-b2 model when training and testing.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> efficientnet_b2 = flowvision.models.efficientnet_b2(pretrained=False, progress=True)

    """
    return _efficientnet(
        "efficientnet_b2", 1.1, 1.2, 0.3, pretrained, progress, **kwargs
    )


@ModelCreator.register_model
def efficientnet_b3(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """
    Constructs the EfficientNet B3 model.

    .. note::
        EfficientNet B3 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
        Note that the (resize-size, crop-size) should be (320, 300) for efficientnet-b3 model when training and testing.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> efficientnet_b3 = flowvision.models.efficientnet_b3(pretrained=False, progress=True)

    """
    return _efficientnet(
        "efficientnet_b3", 1.2, 1.4, 0.3, pretrained, progress, **kwargs
    )


@ModelCreator.register_model
def efficientnet_b4(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """
    Constructs the EfficientNet B4 model.

    .. note::
        EfficientNet B4 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
        Note that the (resize-size, crop-size) should be (384, 380) for efficientnet-b4 model when training and testing.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> efficientnet_b4 = flowvision.models.efficientnet_b4(pretrained=False, progress=True)

    """
    return _efficientnet(
        "efficientnet_b4", 1.4, 1.8, 0.4, pretrained, progress, **kwargs
    )


@ModelCreator.register_model
def efficientnet_b5(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """
    Constructs the EfficientNet B5 model.

    .. note::
        EfficientNet B5 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
        Note that the (resize-size, crop-size) should be (489, 456) for efficientnet-b5 model when training and testing.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> efficientnet_b5 = flowvision.models.efficientnet_b5(pretrained=False, progress=True)

    """
    return _efficientnet(
        "efficientnet_b5",
        1.6,
        2.2,
        0.4,
        pretrained,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


@ModelCreator.register_model
def efficientnet_b6(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """
    Constructs the EfficientNet B6 model.

    .. note::
        EfficientNet B6 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
        Note that the (resize-size, crop-size) should be (561, 528) for efficientnet-b6 model when training and testing.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> efficientnet_b6 = flowvision.models.efficientnet_b6(pretrained=False, progress=True)

    """
    return _efficientnet(
        "efficientnet_b6",
        1.8,
        2.6,
        0.5,
        pretrained,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


@ModelCreator.register_model
def efficientnet_b7(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """
    Constructs the EfficientNet B7 model.

    .. note::
        EfficientNet B7 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
        Note that the (resize-size, crop-size) should be (633, 600) for efficientnet-b7 model when training and testing.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> efficientnet_b7 = flowvision.models.efficientnet_b7(pretrained=False, progress=True)

    """
    return _efficientnet(
        "efficientnet_b7",
        2.0,
        3.1,
        0.5,
        pretrained,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )
