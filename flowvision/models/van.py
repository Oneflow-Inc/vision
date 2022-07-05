"""
Modified from https://github.com/Visual-Attention-Network/VAN-Classification/blob/main/models/van.py
"""

import math
from functools import partial

import oneflow as flow
import oneflow.nn as nn

from flowvision.layers import DropPath, trunc_normal_
from flowvision.models.helpers import to_2tuple
from .utils import load_state_dict_from_url
from .registry import ModelCreator


model_urls = {
    "van_tiny": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VAN/van_tiny.zip",
    "van_small": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VAN/van_small.zip",
    "van_base": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VAN/van_base.zip",
    "van_large": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VAN/van_large.zip",
}


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3
        )
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * flow.ones((dim)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * flow.ones((dim)), requires_grad=True
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        )
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class VAN(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        mlp_ratios=[4, 4, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        num_stages=4,
        flag=False,
    ):
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [
            x.item() for x in flow.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
            )

            block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        mlp_ratio=mlp_ratios[i],
                        drop=drop_rate,
                        drop_path=dpr[cur + j],
                    )
                    for j in range(depths[i])
                ]
            )
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = (
            nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def no_weight_decay(self):
        return {
            "pos_embed1",
            "pos_embed2",
            "pos_embed3",
            "pos_embed4",
            "cls_token",
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


@ModelCreator.register_model
def van_tiny(pretrained: bool = False, progress: bool = True, **kwargs):
    """
    Constructs the VAN-Tiny model trained on ImageNet-1k.

    .. note::
        VAN-Tiny model from `"Visual Attention Network" <https://arxiv.org/pdf/2202.09741.pdf>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> van_tiny = flowvision.models.van_tiny(pretrained=False, progress=True)

    """
    model = VAN(
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 3, 5, 2],
        **kwargs,
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["van_tiny"], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def van_small(pretrained: bool = False, progress: bool = True, **kwargs):
    """
    Constructs the VAN-Small model trained on ImageNet-1k.

    .. note::
        VAN-Small model from `"Visual Attention Network" <https://arxiv.org/pdf/2202.09741.pdf>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> van_small = flowvision.models.van_small(pretrained=False, progress=True)

    """
    model = VAN(
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 4, 2],
        **kwargs,
    )
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["van_small"], progress=progress
        )
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def van_base(pretrained: bool = False, progress: bool = True, **kwargs):
    """
    Constructs the VAN-Base model trained on ImageNet-1k.

    .. note::
        VAN-Base model from `"Visual Attention Network" <https://arxiv.org/pdf/2202.09741.pdf>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> van_base = flowvision.models.van_base(pretrained=False, progress=True)

    """
    model = VAN(
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 3, 12, 3],
        **kwargs,
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["van_base"], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def van_large(pretrained: bool = False, progress: bool = True, **kwargs):
    """
    Constructs the VAN-Large model trained on ImageNet-1k.

    .. note::
        VAN-Large model from `"Visual Attention Network" <https://arxiv.org/pdf/2202.09741.pdf>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> van_large = flowvision.models.van_large(pretrained=False, progress=True)

    """
    model = VAN(
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 5, 27, 3],
        **kwargs,
    )
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["van_large"], progress=progress
        )
        model.load_state_dict(state_dict)
    return model
