"""
Modified from https://github.com/facebookresearch/deit/blob/main/resmlp_models.py
"""
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.init as init
from functools import partial

from flowvision.layers.regularization import DropPath
from flowvision.layers.blocks import Mlp, PatchEmbed

from .utils import load_state_dict_from_url
from .registry import ModelCreator


model_urls = {
    "resmlp_12": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResMLP/resmlp_12.zip",
    "resmlp_12_dist": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResMLP/resmlp_12_dist.zip",
    "resmlp_24": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResMLP/resmlp_24.zip",
    "resmlp_24_dist": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResMLP/resmlp_24_dist.zip",
    "resmlp_24_dino": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResMLP/resmlp_24_dino.zip",
    "resmlp_36": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResMLP/resmlp_36.zip",
    "resmlp_36_dist": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResMLP/resmlp_36_dist.zip",
    "resmlpB_24": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResMLP/resmlpB_24.zip",
    "resmlpB_24_in22k": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResMLP/resmlpB_24_in22k.zip",
    "resmlpB_24_dist": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResMLP/resmlpB_24_dist.zip",
}


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(flow.ones(dim))
        self.beta = nn.Parameter(flow.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta


class layers_scale_mlp_blocks(nn.Module):
    def __init__(
        self,
        dim,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        init_values=1e-4,
        num_patches=196,
    ):
        super().__init__()
        self.norm1 = Affine(dim)
        self.attn = nn.Linear(num_patches, num_patches)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = Affine(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(4.0 * dim),
            act_layer=act_layer,
            drop=drop,
        )
        self.gamma_1 = nn.Parameter(init_values * flow.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * flow.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(
            self.gamma_1 * self.attn(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        )
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class ResMLP(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        drop_rate=0.0,
        Patch_layer=PatchEmbed,
        act_layer=nn.GELU,
        drop_path_rate=0.0,
        init_scale=1e-4,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=int(in_chans),
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        dpr = [drop_path_rate for i in range(depth)]

        self.blocks = nn.ModuleList(
            [
                layers_scale_mlp_blocks(
                    dim=embed_dim,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    act_layer=act_layer,
                    init_values=init_scale,
                    num_patches=num_patches,
                )
                for i in range(depth)
            ]
        )

        self.norm = Affine(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module="head")]
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        x = x.mean(dim=1).reshape(B, 1, -1)  # (B, N, C) -> (B, 1, C)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _create_resmlp(arch, pretrained=False, progress=True, **model_kwargs):
    model = ResMLP(**model_kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def resmlp_12(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ResMLP-12 model.

    .. note::
        ResMLP-12 model from `"ResMLP: Feedforward networks for image classification with data-efficient training" <https://arxiv.org/pdf/2105.03404.pdf>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> resmlp_12 = flowvision.models.resmlp_12(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        Patch_layer=PatchEmbed,
        init_scale=0.1,
        **kwargs
    )
    return _create_resmlp(
        "resmlp_12", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def resmlp_12_dist(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ResMLP-12 model with distillation.

    .. note::
        ResMLP-12 model with distillation from `"ResMLP: Feedforward networks for image classification with data-efficient training" <https://arxiv.org/pdf/2105.03404.pdf>`_.
        Note that tht model is the same as resmlp_12 but the pretrained weight is different.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> resmlp_12_dist = flowvision.models.resmlp_12_dist(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        Patch_layer=PatchEmbed,
        init_scale=0.1,
        **kwargs
    )
    return _create_resmlp(
        "resmlp_12_dist", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def resmlp_24(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ResMLP-24 model.

    .. note::
        ResMLP-24 model from `"ResMLP: Feedforward networks for image classification with data-efficient training" <https://arxiv.org/pdf/2105.03404.pdf>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> resmlp_24 = flowvision.models.resmlp_24(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=24,
        Patch_layer=PatchEmbed,
        init_scale=1e-5,
        **kwargs
    )
    return _create_resmlp(
        "resmlp_24", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def resmlp_24_dist(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ResMLP-24 model with distillation.

    .. note::
        ResMLP-24 model with distillation from `"ResMLP: Feedforward networks for image classification with data-efficient training" <https://arxiv.org/pdf/2105.03404.pdf>`_.
        Note that tht model is the same as resmlp_24 but the pretrained weight is different.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> resmlp_24_dist = flowvision.models.resmlp_24_dist(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=24,
        Patch_layer=PatchEmbed,
        init_scale=1e-5,
        **kwargs
    )
    return _create_resmlp(
        "resmlp_24_dist", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def resmlp_24_dino(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ResMLP-24 model trained under DINO proposed in `"Emerging Properties in Self-Supervised Vision Transformers" <https://arxiv.org/abs/2104.14294>`_.

    .. note::
        ResMLP-24 model with distillation from `"ResMLP: Feedforward networks for image classification with data-efficient training" <https://arxiv.org/pdf/2105.03404.pdf>`_.
        Note that tht model is the same as resmlp_24 but the pretrained weight is different.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> resmlp_24_dino = flowvision.models.resmlp_24_dino(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=24,
        Patch_layer=PatchEmbed,
        init_scale=1e-5,
        **kwargs
    )
    return _create_resmlp(
        "resmlp_24_dino", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def resmlp_36(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ResMLP-36 model.

    .. note::
        ResMLP-36 model from `"ResMLP: Feedforward networks for image classification with data-efficient training" <https://arxiv.org/pdf/2105.03404.pdf>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> resmlp_36 = flowvision.models.resmlp_36(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=36,
        Patch_layer=PatchEmbed,
        init_scale=1e-6,
        **kwargs
    )
    return _create_resmlp(
        "resmlp_36", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def resmlp_36_dist(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ResMLP-36 model with distillation.

    .. note::
        ResMLP-36 model with distillation from `"ResMLP: Feedforward networks for image classification with data-efficient training" <https://arxiv.org/pdf/2105.03404.pdf>`_.
        Note that tht model is the same as resmlp_36 but the pretrained weight is different.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> resmlp_36_dist = flowvision.models.resmlp_36_dist(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=36,
        Patch_layer=PatchEmbed,
        init_scale=1e-6,
        **kwargs
    )
    return _create_resmlp(
        "resmlp_36_dist", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def resmlpB_24(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ResMLP-B-24 model.

    .. note::
        ResMLP-B-24 model from `"ResMLP: Feedforward networks for image classification with data-efficient training" <https://arxiv.org/pdf/2105.03404.pdf>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> resmlpB_24 = flowvision.models.resmlpB_24(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        patch_size=8,
        embed_dim=768,
        depth=24,
        Patch_layer=PatchEmbed,
        init_scale=1e-6,
        **kwargs
    )
    return _create_resmlp(
        "resmlpB_24", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def resmlpB_24_in22k(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ImageNet22k pretrained ResMLP-B-24 model.

    .. note::
        ImageNet22k pretrained ResMLP-B-24 model from `"ResMLP: Feedforward networks for image classification with data-efficient training" <https://arxiv.org/pdf/2105.03404.pdf>`_.
        Note that tht model is the same as resmlpB_24 but the pretrained weight is different.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> resmlpB_24_in22k = flowvision.models.resmlpB_24_in22k(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        patch_size=8,
        embed_dim=768,
        depth=24,
        Patch_layer=PatchEmbed,
        init_scale=1e-6,
        **kwargs
    )
    return _create_resmlp(
        "resmlpB_24_in22k", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def resmlpB_24_dist(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ResMLP-B-24 model with distillation.

    .. note::
        ResMLP-B-24 model with distillation from `"ResMLP: Feedforward networks for image classification with data-efficient training" <https://arxiv.org/pdf/2105.03404.pdf>`_.
        Note that tht model is the same as resmlpB_24 but the pretrained weight is different.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> resmlpB_24_dist = flowvision.models.resmlpB_24_dist(pretrained=False, progress=True)

    """
    model_kwargs = dict(
        patch_size=8,
        embed_dim=768,
        depth=24,
        Patch_layer=PatchEmbed,
        init_scale=1e-6,
        **kwargs
    )
    return _create_resmlp(
        "resmlpB_24_dist", pretrained=pretrained, progress=progress, **model_kwargs
    )
