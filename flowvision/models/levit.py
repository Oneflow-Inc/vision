"""
Modified from https://github.com/facebookresearch/LeViT
"""
import itertools

import oneflow as flow

from flowvision.layers import trunc_normal_
from .utils import load_state_dict_from_url
from .registry import ModelCreator


model_urls = {
    "levit_128s": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/LeViT/levit_128s.zip",
    "levit_128": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/LeViT/levit_128.zip",
    "levit_192": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/LeViT/levit_192.zip",
    "levit_256": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/LeViT/levit_256.zip",
    "levit_384": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/LeViT/levit_384.zip",
}


class Conv2d_BN(flow.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', flow.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = flow.nn.BatchNorm2d(b)
        flow.nn.init.constant_(bn.weight, bn_weight_init)
        flow.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Linear_BN(flow.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', flow.nn.Linear(a, b, bias=False))
        bn = flow.nn.BatchNorm1d(b)
        flow.nn.init.constant_(bn.weight, bn_weight_init)
        flow.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape(*x.shape)


class BN_Linear(flow.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', flow.nn.BatchNorm1d(a))
        l = flow.nn.Linear(a, b, bias=bias)
        trunc_normal_(l.weight, std=std)
        if bias:
            flow.nn.init.constant_(l.bias, 0)
        self.add_module('l', l)


def b16(n, activation, resolution=224):
    return flow.nn.Sequential(
        Conv2d_BN(3, n // 8, 3, 2, 1, resolution=resolution),
        activation(),
        Conv2d_BN(n // 8, n // 4, 3, 2, 1, resolution=resolution // 2),
        activation(),
        Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=resolution // 4),
        activation(),
        Conv2d_BN(n // 2, n, 3, 2, 1, resolution=resolution // 8))


class Residual(flow.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * (flow.rand(x.size(0), 1, 1, device=x.device)>self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class Attention(flow.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, h, resolution=resolution)
        self.proj = flow.nn.Sequential(activation(), Linear_BN(
            self.dh, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = flow.nn.Parameter(
            flow.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             flow.LongTensor(idxs).view(N, N))

    @flow.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x

class Subsample(flow.nn.Module):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[
            :, ::self.stride, ::self.stride].reshape(B, -1, C)
        return x


class AttentionSubsample(flow.nn.Module):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution=14, resolution_=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_**2
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h, resolution=resolution)

        self.q = flow.nn.Sequential(
            Subsample(stride, resolution),
            Linear_BN(in_dim, nh_kd, resolution=resolution_))
        self.proj = flow.nn.Sequential(activation(), Linear_BN(
            self.dh, out_dim, resolution=resolution_))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(
            range(resolution_), range(resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = flow.nn.Parameter(
            flow.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             flow.LongTensor(idxs).view(N_, N))

    @flow.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -
                               1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, self.resolution_2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


class LeViT(flow.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 mlp_ratio=[2],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=flow.nn.Hardswish,
                 mlp_activation=flow.nn.Hardswish,
                 distillation=True,
                 drop_path=0):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation

        self.patch_embed = hybrid_backbone

        self.blocks = []
        down_ops.append([''])
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                    ), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(flow.nn.Sequential(
                            Linear_BN(ed, h, resolution=resolution),
                            mlp_activation(),
                            Linear_BN(h, ed, bn_weight_init=0,
                                      resolution=resolution),
                        ), drop_path))
            if do[0] == 'Subsample':
                #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(flow.nn.Sequential(
                            Linear_BN(embed_dim[i + 1], h,
                                      resolution=resolution),
                            mlp_activation(),
                            Linear_BN(
                                h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution),
                        ), drop_path))
        self.blocks = flow.nn.Sequential(*self.blocks)

        # Classifier head
        self.head = BN_Linear(
            embed_dim[-1], num_classes) if num_classes > 0 else flow.nn.Identity()
        if distillation:
            self.head_dist = BN_Linear(
                embed_dim[-1], num_classes) if num_classes > 0 else flow.nn.Identity()


    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        x = x.mean(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x

def model_factory(C, D, X, N, drop_path,
                  num_classes, distillation, pretrained, name, progress=True):
    embed_dim = [int(x) for x in C.split('_')]
    num_heads = [int(x) for x in N.split('_')]
    depth = [int(x) for x in X.split('_')]
    act = flow.nn.Hardswish
    model = LeViT(
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[D] * 3,
        depth=depth,
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', D, embed_dim[0] // D, 4, 2, 2],
            ['Subsample', D, embed_dim[1] // D, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act),
        num_classes=num_classes,
        drop_path=drop_path,
        distillation=distillation
    )
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[name], progress=progress
        )
        model.load_state_dict(state_dict)

    return model


specification = {
    'levit_128s': {'C': '128_256_384', 'D': 16, 'N': '4_6_8', 'X': '2_3_4', 'drop_path': 0},
    'levit_128': {'C': '128_256_384', 'D': 16, 'N': '4_8_12', 'X': '4_4_4', 'drop_path': 0},
    'levit_192': {'C': '192_288_384', 'D': 32, 'N': '3_5_6', 'X': '4_4_4', 'drop_path': 0},
    'levit_256': {'C': '256_384_512', 'D': 32, 'N': '4_6_8', 'X': '4_4_4', 'drop_path': 0},
    'levit_384': {'C': '384_512_768', 'D': 32, 'N': '6_9_12', 'X': '4_4_4', 'drop_path': 0.1},
}


@ModelCreator.register_model
def levit_128s(num_classes=1000, distillation=True,
               pretrained=False):
    """
    Constructs the LeViT-128S model.

    .. note::
        LeViT-128S model architecture from the `LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference <https://arxiv.org/abs/2104.01136>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> levit_128s = flowvision.models.levit_128s(pretrained=False, progress=True)

    """
    return model_factory(**specification['levit_128s'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, name="levit_128s")

@ModelCreator.register_model
def levit_128(num_classes=1000, distillation=True,
               pretrained=False):
    """
    Constructs the LeViT-128 model.

    .. note::
        LeViT-128 model architecture from the `LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference <https://arxiv.org/abs/2104.01136>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> levit_128 = flowvision.models.levit_128(pretrained=False, progress=True)

    """
    return model_factory(**specification['levit_128'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, name="levit_128")

@ModelCreator.register_model
def levit_192(num_classes=1000, distillation=True,
               pretrained=False):
    """
    Constructs the LeViT-192 model.

    .. note::
        LeViT-192 model architecture from the `LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference <https://arxiv.org/abs/2104.01136>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> levit_192 = flowvision.models.levit_192(pretrained=False, progress=True)

    """
    return model_factory(**specification['levit_192'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, name="levit_192")

@ModelCreator.register_model
def levit_256(num_classes=1000, distillation=True,
               pretrained=False):
    """
    Constructs the LeViT-256 model.

    .. note::
        LeViT-256 model architecture from the `LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference <https://arxiv.org/abs/2104.01136>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> levit_256 = flowvision.models.levit_256(pretrained=False, progress=True)

    """
    return model_factory(**specification['levit_256'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, name="levit_256")

@ModelCreator.register_model
def levit_384(num_classes=1000, distillation=True,
               pretrained=False):
    """
    Constructs the LeViT-384 model.

    .. note::
        LeViT-384 model architecture from the `LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference <https://arxiv.org/abs/2104.01136>`_ paper.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> levit_384 = flowvision.models.levit_384(pretrained=False, progress=True)

    """
    return model_factory(**specification['levit_384'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, name="levit_384")
