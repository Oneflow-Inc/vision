import os

import oneflow as flow
import oneflow.nn as nn
from oneflow import Tensor
from typing import Type, Any, Callable, Union, List, Optional


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
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        fuse_bn_relu=False,
        fuse_bn_add_relu=False,
    ) -> None:
        super(Bottleneck, self).__init__()
        self.fuse_bn_relu = fuse_bn_relu
        self.fuse_bn_add_relu = fuse_bn_add_relu
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)

        if self.fuse_bn_relu:
            self.bn1 = nn.FusedBatchNorm2d(width)
            self.bn2 = nn.FusedBatchNorm2d(width)
        else:
            self.bn1 = norm_layer(width)
            self.bn2 = norm_layer(width)
            self.relu = nn.ReLU()

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.conv3 = conv1x1(width, planes * self.expansion)

        if self.fuse_bn_add_relu:
            self.bn3 = nn.FusedBatchNorm2d(planes * self.expansion)
        else:
            self.bn3 = norm_layer(planes * self.expansion)
            self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsample is not None:
            # Note self.downsample execute before  self.conv1 has better performance
            # when open allow_fuse_add_to_output optimizatioin in nn.Graph.
            # Reference: https://github.com/Oneflow-Inc/OneTeam/issues/840#issuecomment-994903466
            # Reference: https://github.com/NVIDIA/cudnn-frontend/issues/21
            identity = self.downsample(x)

        out = self.conv1(x)

        if self.fuse_bn_relu:
            out = self.bn1(out, None)
        else:
            out = self.bn1(out)
            out = self.relu(out)

        out = self.conv2(out)

        if self.fuse_bn_relu:
            out = self.bn2(out, None)
        else:
            out = self.bn2(out)
            out = self.relu(out)

        out = self.conv3(out)

        if self.fuse_bn_add_relu:
            out = self.bn3(out, identity)
        else:
            out = self.bn3(out)
            out += identity
            out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        fuse_bn_relu=False,
        fuse_bn_add_relu=False,
        channel_last=False,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.fuse_bn_relu = fuse_bn_relu
        self.fuse_bn_add_relu = fuse_bn_add_relu
        self.channel_last = channel_last
        if self.channel_last:
            self.pad_input = True
        else:
            self.pad_input = False

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        if self.pad_input:
            channel_size = 4
        else:
            channel_size = 3
        if self.channel_last:
            os.environ["ONEFLOW_ENABLE_NHWC"] = "1"
        self.conv1 = nn.Conv2d(
            channel_size, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )

        if self.fuse_bn_relu:
            self.bn1 = nn.FusedBatchNorm2d(self.inplanes)
        else:
            self.bn1 = self._norm_layer(self.inplanes)
            self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AvgPool2d((7, 7), stride=(1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                fuse_bn_relu=self.fuse_bn_relu,
                fuse_bn_add_relu=self.fuse_bn_add_relu,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    fuse_bn_relu=self.fuse_bn_relu,
                    fuse_bn_add_relu=self.fuse_bn_add_relu,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        if self.pad_input:
            if self.channel_last:
                # NHWC
                paddings = (0, 1)
            else:
                # NCHW
                paddings = (0, 0, 0, 0, 0, 1)
            x = flow._C.pad(x, pad=paddings, mode="constant", value=0)
        x = self.conv1(x)
        if self.fuse_bn_relu:
            x = self.bn1(x, None)
        else:
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet50(**kwargs: Any) -> ResNet:
    r"""ResNet-5
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], **kwargs)