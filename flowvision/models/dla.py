"""
Modified from https://github.com/ucbdrive/dla/blob/master/dla.py
"""
import math

import oneflow as flow
from oneflow import nn

from .utils import load_state_dict_from_url
from .registry import ModelCreator

BatchNorm = nn.BatchNorm2d


model_urls = {
    "dla34": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/DLA/dla34-ba72cf86.zip",
    "dla46_c": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/DLA/dla46_c-2bfd52c3.zip",
    "dla46x_c": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/DLA/dla46x_c-d761bae7.zip",
    "dla60x_c": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/DLA/dla60x_c-b870c45c.zip",
    "dla60": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/DLA/dla60-24839fc4.zip",
    "dla60x": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/DLA/dla60x-d15cacda.zip",
    "dla102": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/DLA/dla102-d94d9790.zip",
    "dla102x": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/DLA/dla102x-ad62be81.zip",
    "dla102x2": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/DLA/dla102x2-262837b6.zip",
    "dla169": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/DLA/dla169-0914e092.zip",
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
            groups=cardinality,
        )
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(flow.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(
        self,
        levels,
        block,
        in_channels,
        out_channels,
        stride=1,
        level_root=False,
        root_dim=0,
        root_kernel_size=1,
        dilation=1,
        root_residual=False,
    ):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                ),
                BatchNorm(out_channels),
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(
        self,
        levels,
        channels,
        num_classes=1000,
        block=BasicBlock,
        residual_root=False,
        return_levels=False,
        pool_size=7,
        linear_root=False,
    ):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2
        )
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root,
        )
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root,
        )

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(
            channels[-1], num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
                BatchNorm(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes,
                        kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                    BatchNorm(planes),
                    nn.ReLU(inplace=True),
                ]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, "level{}".format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            return x


@ModelCreator.register_model
def dla34(pretrained=False, progress=True, **kwargs):
    """
    Constructs DLA34 224x224 model trained on ImageNet-1k.

    .. note::
        DLA34 224x224 model from `"Deep Layer Aggregation" <https://arxiv.org/pdf/1707.06484>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> dla34 = flowvision.models.dla34(pretrained=False, progress=True)

    """
    model = DLA(
        [1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["dla34"], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def dla46_c(pretrained=False, progress=True, **kwargs):
    """
    Constructs DLA46_c 224x224 model trained on ImageNet-1k.

    .. note::
        DLA46_c 224x224 model from `"Deep Layer Aggregation" <https://arxiv.org/pdf/1707.06484>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> dla46_c = flowvision.models.dla46_c(pretrained=False, progress=True)

    """
    Bottleneck.expansion = 2
    model = DLA(
        [1, 1, 1, 2, 2, 1], [16, 32, 64, 64, 128, 256], block=Bottleneck, **kwargs
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["dla46_c"], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def dla46x_c(pretrained=False, progress=True, **kwargs):
    """
    Constructs DLA46x_c 224x224 model trained on ImageNet-1k.

    .. note::
        DLA46x_c 224x224 model from `"Deep Layer Aggregation" <https://arxiv.org/pdf/1707.06484>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> dla46x_c = flowvision.models.dla46x_c(pretrained=False, progress=True)

    """
    BottleneckX.expansion = 2
    model = DLA(
        [1, 1, 1, 2, 2, 1], [16, 32, 64, 64, 128, 256], block=BottleneckX, **kwargs
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["dla46x_c"], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def dla60x_c(pretrained=False, progress=True, **kwargs):
    """
    Constructs DLA60x_c 224x224 model trained on ImageNet-1k.

    .. note::
        DLA60x_c 224x224 model from `"Deep Layer Aggregation" <https://arxiv.org/pdf/1707.06484>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> dla60x_c = flowvision.models.dla60x_c(pretrained=False, progress=True)

    """
    BottleneckX.expansion = 2
    model = DLA(
        [1, 1, 1, 2, 3, 1], [16, 32, 64, 64, 128, 256], block=BottleneckX, **kwargs
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["dla60x_c"], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def dla60(pretrained=False, progress=True, **kwargs):
    """
    Constructs DLA60 224x224 model trained on ImageNet-1k.

    .. note::
        DLA60 224x224 model from `"Deep Layer Aggregation" <https://arxiv.org/pdf/1707.06484>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> dla60 = flowvision.models.dla60(pretrained=False, progress=True)

    """
    Bottleneck.expansion = 2
    model = DLA(
        [1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024], block=Bottleneck, **kwargs
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["dla60"], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def dla60x(pretrained=False, progress=True, **kwargs):
    """
    Constructs DLA60x 224x224 model trained on ImageNet-1k.

    .. note::
        DLA60x 224x224 model from `"Deep Layer Aggregation" <https://arxiv.org/pdf/1707.06484>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> dla60x = flowvision.models.dla60x(pretrained=False, progress=True)

    """
    BottleneckX.expansion = 2
    model = DLA(
        [1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024], block=BottleneckX, **kwargs
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["dla60x"], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def dla102(pretrained=False, progress=True, **kwargs):
    """
    Constructs DLA102 224x224 model trained on ImageNet-1k.

    .. note::
        DLA102 224x224 model from `"Deep Layer Aggregation" <https://arxiv.org/pdf/1707.06484>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> dla102 = flowvision.models.dla102(pretrained=False, progress=True)

    """
    Bottleneck.expansion = 2
    model = DLA(
        [1, 1, 1, 3, 4, 1],
        [16, 32, 128, 256, 512, 1024],
        block=Bottleneck,
        residual_root=True,
        **kwargs
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["dla102"], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def dla102x(pretrained=False, progress=True, **kwargs):
    """
    Constructs DLA102x 224x224 model trained on ImageNet-1k.

    .. note::
        DLA102x 224x224 model from `"Deep Layer Aggregation" <https://arxiv.org/pdf/1707.06484>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> dla102x = flowvision.models.dla102x(pretrained=False, progress=True)

    """
    BottleneckX.expansion = 2
    model = DLA(
        [1, 1, 1, 3, 4, 1],
        [16, 32, 128, 256, 512, 1024],
        block=BottleneckX,
        residual_root=True,
        **kwargs
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["dla102x"], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def dla102x2(pretrained=False, progress=True, **kwargs):
    """
    Constructs DLA102x2 224x224 model trained on ImageNet-1k.

    .. note::
        DLA102x2 224x224 model from `"Deep Layer Aggregation" <https://arxiv.org/pdf/1707.06484>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> dla102x2 = flowvision.models.dla102x2(pretrained=False, progress=True)

    """
    BottleneckX.cardinality = 64
    model = DLA(
        [1, 1, 1, 3, 4, 1],
        [16, 32, 128, 256, 512, 1024],
        block=BottleneckX,
        residual_root=True,
        **kwargs
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["dla102x2"], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def dla169(pretrained=False, progress=True, **kwargs):
    """
    Constructs DLA169 224x224 model trained on ImageNet-1k.

    .. note::
        DLA169 224x224 model from `"Deep Layer Aggregation" <https://arxiv.org/pdf/1707.06484>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> dla169 = flowvision.models.dla169(pretrained=False, progress=True)

    """
    Bottleneck.expansion = 2
    model = DLA(
        [1, 1, 2, 3, 5, 1],
        [16, 32, 128, 256, 512, 1024],
        block=Bottleneck,
        residual_root=True,
        **kwargs
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["dla169"], progress=progress)
        model.load_state_dict(state_dict)
    return model
