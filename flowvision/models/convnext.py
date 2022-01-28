"""
Modified from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
"""

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

from flowvision.layers.weight_init import trunc_normal_
from flowvision.layers.regularization import DropPath
from .registry import ModelCreator
from .utils import load_state_dict_from_url


model_urls = {
    "convnext_tiny_224": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ConvNeXt/convnext_tiny_1k_224_ema.zip",
    "convnext_small_224": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_224": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_base_384": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth",
    "convnext_large_224": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_large_384": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth",
    "convnext_base_224_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_base_224_22k_to_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth",
    "convnext_base_384_22k_to_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth",
    "convnext_large_224_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_large_224_22k_to_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth",
    "convnext_large_384_22k_to_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth",
    "convnext_xlarge_224_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
    "convnext_xlarge_224_to_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth",
    "convnext_xlarge_384_to_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth",
    "convnext_iso_small_224": "https://dl.fbaipublicfiles.com/convnext/convnext_iso_small_1k_224_ema.pth",
    "convnext_iso_base_224": "https://dl.fbaipublicfiles.com/convnext/convnext_iso_base_1k_224_ema.pth",
    "convnext_iso_large_224": "https://dl.fbaipublicfiles.com/convnext/convnext_iso_large_1k_224_ema.pth",
}


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * flow.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNext(nn.Module):
    r""" ConvNeXt
        The OneFlow impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, 
                 in_chans=3, 
                 num_classes=1000, 
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.,):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in flow.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ConvNextIsotropic(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self,
                 in_chans=3, 
                 num_classes=1000,
                 depth=18,
                 dim=384,
                 drop_path_rate=0.,
                 layer_scale_init_value=0,
                 head_init_scale=1.):
        super().__init__()

        self.stem = nn.Conv2d(in_chans, dim, kernel_size=16, stride=16)
        dp_rates=[x.item() for x in flow.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.Sequential(*[Block(dim=dim, drop_path=dp_rates[i], 
                                    layer_scale_init_value=layer_scale_init_value)
                                    for i in range(depth)])

        self.norm = LayerNorm(dim, eps=1e-6) # final norm layer
        self.head = nn.Linear(dim, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(flow.ones(normalized_shape))
        self.bias = nn.Parameter(flow.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            # FIXME: use F.layer_norm
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / flow.sqrt(s + self.eps)
            x = self.weight[None, None, :] * x + self.bias[None, None, :]
            return x

        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / flow.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def _create_convnext(arch, pretrained=False, progress=True, **model_kwargs):
    model = ConvNext(**model_kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def _create_convnext_isotropic(arch, pretrained=False, progress=True, **model_kwargs):
    model = ConvNextIsotropic(**model_kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def convnext_tiny_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ConvNext-Tiny model trained on ImageNet2012

    .. note::
        ConvNext-Tiny model from `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>` _.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> convnext_tiny_224 = flowvision.models.convnext_tiny_224(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        **kwargs
    )
    return _create_convnext(
        "convnext_tiny_224", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def convnext_small_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ConvNext-Small model trained on ImageNet2012

    .. note::
        ConvNext-Small model from `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>` _.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> convnext_small_224 = flowvision.models.convnext_small_224(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        **kwargs
    )
    return _create_convnext(
        "convnext_small_224", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def convnext_base_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ConvNext-Base model trained on ImageNet2012

    .. note::
        ConvNext-Base model from `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>` _.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> convnext_base_224 = flowvision.models.convnext_base_224(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        **kwargs
    )
    return _create_convnext(
        "convnext_base_224", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def convnext_large_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ConvNext-Large model trained on ImageNet2012

    .. note::
        ConvNext-Large model from `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>` _.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> convnext_large_224 = flowvision.models.convnext_large_224(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        **kwargs
    )
    return _create_convnext(
        "convnext_large_224", pretrained=pretrained, progress=progress, **model_kwargs
    )