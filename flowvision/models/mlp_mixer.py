import math
from functools import partial

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.init as init

from .utils import load_state_dict_from_url, named_apply
from .registry import ModelCreator
from flowvision.layers.regularization import DropPath
from flowvision.layers.blocks import PatchEmbed
from flowvision.layers.weight_init import lecun_normal_


model_urls = {
    "mlp_mixer_s16_224": None,
    "mlp_mixer_s32_224": None,
    "mlp_mixer_b16_224": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/Mlp-Mixer/mlp_mixer_b16_224.zip",
    "mlp_mixer_b32_224": None,
    "mlp_mixer_b16_224_in21k": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/Mlp-Mixer/mlp_mixer_b16_224_in21k.zip",
    "mlp_mixer_l16_224": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/Mlp-Mixer/mlp_mixer_l16_224.zip",
    "mlp_mixer_l32_224": None,
    "mlp_mixer_l16_224_in21k": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/Mlp-Mixer/mlp_mixer_l16_224_in21k.zip",
    "mlp_mixer_b16_224_miil": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/Mlp-Mixer/mlp_mixer_b16_224_miil.zip",
    "mlp_mixer_b16_224_miil_in21k": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/Mlp-Mixer/mlp_mixer_b16_224_miil_in21k.zip",
    "gmlp_ti16_224": None,
    "gmlp_s16_224": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/Mlp-Mixer/gmlp_s16_224.zip",
    "gmlp_b16_224": None,
}


# helpers
def pair(x):
    if not isinstance(x, tuple):
        return (x, x)
    else:
        return x


class Mlp(nn.Module):
    """
    You can also import Mlp Block in flowvision.layers.blocks like this:
    from flowvision.layers.blocks import Mlp
    """

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
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        gate_layer=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2
        else:
            self.gate = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SpatialGatingUnit(nn.Module):
    """ Spatial Gating Unit
    """

    def __init__(self, dim, num_patches, norm_layer=nn.LayerNorm):
        super().__init__()
        gate_dim = dim // 2
        self.norm = norm_layer(gate_dim)
        self.proj = nn.Linear(num_patches, num_patches)

    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        nn.init.normal_(self.proj.weight, std=1e-6)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        # TODO: use fixed chunk op
        # u, v = x.chunk(2, dim=-1)
        B, N, C = x.size()
        split_dim = C // 2
        u, v = flow.split(x, split_dim, dim=-1)[0], flow.split(x, split_dim, dim=-1)[1]
        v = self.norm(v)
        v = self.proj(v.transpose(-1, -2))
        return u * v.transpose(-1, -2)


class SpatialGatingBlock(nn.Module):
    """ Residual Block w/ Spatial Gating
    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """

    def __init__(
        self,
        dim,
        num_patches,
        mlp_ratio=4,
        mlp_layer=GatedMlp,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm = norm_layer(dim)
        sgu = partial(SpatialGatingUnit, num_patches=num_patches)
        self.mlp_channels = mlp_layer(
            dim, channel_dim, act_layer=act_layer, gate_layer=sgu, drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """

    def __init__(
        self,
        dim,
        num_patches,
        mlp_ratio=(0.5, 4.0),
        mlp_layer=Mlp,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in pair(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(
            num_patches, tokens_dim, act_layer=act_layer, drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(
            self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        )
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class MlpMixer(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        img_size=224,
        in_chans=3,
        patch_size=16,
        num_blocks=8,
        embed_dim=512,
        mlp_ratio=(0.5, 4.0),
        block_layer=MixerBlock,
        mlp_layer=Mlp,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        drop_rate=0.0,
        drop_path_rate=0.0,
        nlhb=False,
        stem_norm=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models

        self.stem = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if stem_norm else None,
        )
        # TODO consistent the drop-path-rate rule with the original repo
        self.blocks = nn.Sequential(
            *[
                block_layer(
                    embed_dim,
                    self.stem.num_patches,
                    mlp_ratio,
                    mlp_layer=mlp_layer,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    drop=drop_rate,
                    drop_path=drop_path_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        )

        self.init_weights(nlhb=nlhb)

    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.0
        named_apply(
            partial(_init_weights, head_bias=head_bias), module=self
        )  # depth-first

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_weights(module: nn.Module, name: str, head_bias: float = 0.0, flax=False):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # like MLP init in vit
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if "mlp" in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in depth-first order.
        module.init_weights()


def _create_mlp_mixer(arch, pretrained=False, progress=True, **model_kwargs):
    model = MlpMixer(**model_kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def mlp_mixer_s16_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Mixer-S/16 224x224 model.

    .. note::
        Mixer-S/16 224x224 model from `"MLP-Mixer: An all-MLP Architecture for Vision" <https://arxiv.org/pdf/2105.01601.pdf>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mlp_mixer_s16_224 = flowvision.models.mlp_mixer_s16_224(pretrained=False, progress=True)

    """
    model_kwargs = dict(patch_size=16, num_blocks=8, embed_dim=512, **kwargs)
    return _create_mlp_mixer(
        "mlp_mixer_s16_224", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def mlp_mixer_s32_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Mixer-S/32 224x224 model.

    .. note::
        Mixer-S/32 224x224 model from `"MLP-Mixer: An all-MLP Architecture for Vision" <https://arxiv.org/pdf/2105.01601.pdf>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mlp_mixer_s32_224 = flowvision.models.mlp_mixer_s32_224(pretrained=False, progress=True)

    """
    model_kwargs = dict(patch_size=32, num_blocks=8, embed_dim=512, **kwargs)
    return _create_mlp_mixer(
        "mlp_mixer_s32_224", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def mlp_mixer_b16_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Mixer-B/16 224x224 model.

    .. note::
        Mixer-B/16 224x224 model from `"MLP-Mixer: An all-MLP Architecture for Vision" <https://arxiv.org/pdf/2105.01601.pdf>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mlp_mixer_b16_224 = flowvision.models.mlp_mixer_b16_224(pretrained=False, progress=True)

    """
    model_kwargs = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    return _create_mlp_mixer(
        "mlp_mixer_b16_224", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def mlp_mixer_b32_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Mixer-B/32 224x224 model.

    .. note::
        Mixer-B/32 224x224 model from `"MLP-Mixer: An all-MLP Architecture for Vision" <https://arxiv.org/pdf/2105.01601.pdf>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mlp_mixer_b32_224 = flowvision.models.mlp_mixer_b32_224(pretrained=False, progress=True)

    """
    model_kwargs = dict(patch_size=32, num_blocks=12, embed_dim=768, **kwargs)
    return _create_mlp_mixer(
        "mlp_mixer_b32_224", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def mlp_mixer_b16_224_in21k(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Mixer-B/16 224x224 ImageNet21k pretrained model.

    .. note::
        Mixer-B/16 224x224 ImageNet21k pretrained model from `"MLP-Mixer: An all-MLP Architecture for Vision" <https://arxiv.org/pdf/2105.01601.pdf>`_.
        Note that this model is the pretrained model for fine-tune on different datasets.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mlp_mixer_b16_224_in21k = flowvision.models.mlp_mixer_b16_224_in21k(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        num_classes=21843, patch_size=16, num_blocks=12, embed_dim=768, **kwargs
    )
    return _create_mlp_mixer(
        "mlp_mixer_b16_224_in21k",
        pretrained=pretrained,
        progress=progress,
        **model_kwargs
    )


@ModelCreator.register_model
def mlp_mixer_l16_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Mixer-L/16 224x224 model.

    .. note::
        Mixer-L/16 224x224 model from `"MLP-Mixer: An all-MLP Architecture for Vision" <https://arxiv.org/pdf/2105.01601.pdf>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mlp_mixer_l16_224 = flowvision.models.mlp_mixer_l16_224(pretrained=False, progress=True)

    """
    model_kwargs = dict(patch_size=16, num_blocks=24, embed_dim=1024, **kwargs)
    return _create_mlp_mixer(
        "mlp_mixer_l16_224", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def mlp_mixer_l32_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Mixer-L/32 224x224 model.

    .. note::
        Mixer-L/32 224x224 model from `"MLP-Mixer: An all-MLP Architecture for Vision" <https://arxiv.org/pdf/2105.01601.pdf>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mlp_mixer_l32_224 = flowvision.models.mlp_mixer_l32_224(pretrained=False, progress=True)

    """
    model_kwargs = dict(patch_size=32, num_blocks=24, embed_dim=1024, **kwargs)
    return _create_mlp_mixer(
        "mlp_mixer_l32_224", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def mlp_mixer_l16_224_in21k(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Mixer-L/16 224x224 ImageNet21k pretrained model.

    .. note::
        Mixer-L/16 224x224 ImageNet21k pretrained model from `"MLP-Mixer: An all-MLP Architecture for Vision" <https://arxiv.org/pdf/2105.01601.pdf>`_.
        Note that this model is the pretrained model for fine-tune on different datasets.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mlp_mixer_l16_224_in21k = flowvision.models.mlp_mixer_l16_224_in21k(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        num_classes=21843, patch_size=16, num_blocks=24, embed_dim=1024, **kwargs
    )
    return _create_mlp_mixer(
        "mlp_mixer_l16_224_in21k",
        pretrained=pretrained,
        progress=progress,
        **model_kwargs
    )


@ModelCreator.register_model
def mlp_mixer_b16_224_miil(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Mixer-B/16 224x224 model with different weights.

    .. note::
        Mixer-B/16 224x224 model from `"MLP-Mixer: An all-MLP Architecture for Vision" <https://arxiv.org/pdf/2105.01601.pdf>`_.
        Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mlp_mixer_b16_224_miil = flowvision.models.mlp_mixer_b16_224_miil(pretrained=False, progress=True)

    """
    model_kwargs = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    return _create_mlp_mixer(
        "mlp_mixer_b16_224_miil",
        pretrained=pretrained,
        progress=progress,
        **model_kwargs
    )


@ModelCreator.register_model
def mlp_mixer_b16_224_miil_in21k(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Mixer-B/16 224x224 ImageNet21k pretrained model.

    .. note::
        Mixer-B/16 224x224 ImageNet21k pretrained model from `"MLP-Mixer: An all-MLP Architecture for Vision" <https://arxiv.org/pdf/2105.01601.pdf>`_.
        Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mlp_mixer_b16_224_miil_in21k = flowvision.models.mlp_mixer_b16_224_miil_in21k(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        num_classes=11221, patch_size=16, num_blocks=12, embed_dim=768, **kwargs
    )
    return _create_mlp_mixer(
        "mlp_mixer_b16_224_miil_in21k",
        pretrained=pretrained,
        progress=progress,
        **model_kwargs
    )


@ModelCreator.register_model
def gmlp_ti16_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the gMLP-tiny-16 224x224 model.

    .. note::
        gMLP-tiny-16 224x224 model from `"Pay Attention to MLPs" <https://arxiv.org/pdf/2105.08050.pdf>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> gmlp_ti16_224 = flowvision.models.gmlp_ti16_224(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        patch_size=16,
        num_blocks=30,
        embed_dim=128,
        mlp_ratio=6,
        block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp,
        **kwargs
    )
    return _create_mlp_mixer(
        "gmlp_ti16_224", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def gmlp_s16_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the gMLP-small-16 224x224 model.

    .. note::
        gMLP-small-16 224x224 model from `"Pay Attention to MLPs" <https://arxiv.org/pdf/2105.08050.pdf>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> gmlp_s16_224 = flowvision.models.gmlp_s16_224(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        patch_size=16,
        num_blocks=30,
        embed_dim=256,
        mlp_ratio=6,
        block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp,
        **kwargs
    )
    return _create_mlp_mixer(
        "gmlp_s16_224", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def gmlp_b16_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the gMLP-base-16 224x224 model.

    .. note::
        gMLP-base-16 224x224 model from `"Pay Attention to MLPs" <https://arxiv.org/pdf/2105.08050.pdf>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> gmlp_b16_224 = flowvision.models.gmlp_b16_224(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        patch_size=16,
        num_blocks=30,
        embed_dim=512,
        mlp_ratio=6,
        block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp,
        **kwargs
    )
    return _create_mlp_mixer(
        "gmlp_b16_224", pretrained=pretrained, progress=progress, **model_kwargs
    )
