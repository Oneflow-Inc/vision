"""
Modified from https://github.com/apple/ml-cvnets/blob/d38a116fe134a8cd5db18670764fdaafd39a5d4f/cvnets/models/classification/mobilevit.py
"""
import math
from typing import Dict, Tuple, Optional, Union

import oneflow as flow
import oneflow.nn as nn
from oneflow import Tensor
from oneflow.nn import functional as F

from .registry import ModelCreator
from .utils import load_state_dict_from_url


model_urls = {
    "mobilevit_small": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/MobileViT/mobilevit_s.zip",
    "mobilevit_x_small": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/MobileViT/mobilevit_xs.zip",
    "mobilevit_xx_small": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/MobileViT/mobilevit_xxs.zip"
}


class MultiHeadAttention(nn.Module):
    """
        This layer applies a multi-head attention as described in "Attention is all you need" paper
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: Optional[float] = 0.0,
                 bias: Optional[bool] = True):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        :param attn_dropout: Attention dropout
        :param bias: Bias
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Got: embed_dim={} and num_heads={}".format(embed_dim, num_heads)

        self.qkv_proj = LinearLayer(in_features=embed_dim, out_features=3 * embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = LinearLayer(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        # [B x N x C]
        b_sz, n_patches, in_channels = x.shape

        # [B x N x C] --> [B x N x 3 x h x C]
        qkv = (
            self.qkv_proj(x)
                .reshape(b_sz, n_patches, 3, self.num_heads, -1)
        )
        # [B x N x 3 x h x C] --> [B x h x 3 x N x C]
        qkv = qkv.transpose(1, 3)

        # [B x h x 3 x N x C] --> [B x h x N x C] x 3
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query * self.scaling

        # [B x h x N x C] --> [B x h x c x N]
        key = key.transpose(2, 3)

        # QK^T
        # [B x h x N x c] x [B x h x c x N] --> [B x h x N x N]
        attn = flow.matmul(query, key)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [B x h x N x N] x [B x h x N x c] --> [B x h x N x c]
        out = flow.matmul(attn, value)

        # [B x h x N x c] --> [B x N x h x c] --> [B x N x C=ch]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)
        out = self.out_proj(out)

        return out


class TransformerEncoder(nn.Module):
    """
        This class defines the Transformer encoder (pre-norm) as described in "Attention is all you need" paper
            https://arxiv.org/abs/1706.03762
    """

    def __init__(self, embed_dim: int, ffn_latent_dim: int, num_heads: Optional[int] = 8,
                 attn_dropout: Optional[float] = 0.0,
                 dropout: Optional[float] = 0.1, ffn_dropout: Optional[float] = 0.0):
        super(TransformerEncoder, self).__init__()

        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim),
            MultiHeadAttention(embed_dim, num_heads, attn_dropout=attn_dropout, bias=True),
            nn.Dropout(dropout)
        )

        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            LinearLayer(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(ffn_dropout),
            LinearLayer(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(dropout)
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout

    def forward(self, x: Tensor) -> Tensor:
        # Multi-head attention
        x = x + self.pre_norm_mha(x)

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class MobileViTBlock(nn.Module):
    """
        MobileViT block: https://arxiv.org/abs/2110.02178?context=cs.LG
    """

    def __init__(self, in_channels: int, transformer_dim: int, ffn_dim: int,
                 n_transformer_blocks: Optional[int] = 2,
                 head_dim: Optional[int] = 32, attn_dropout: Optional[float] = 0.1,
                 dropout: Optional[float] = 0.1, ffn_dropout: Optional[float] = 0.1, patch_h: Optional[int] = 8,
                 patch_w: Optional[int] = 8,
                 conv_ksize: Optional[int] = 3,
                 dilation: Optional[int] = 1, var_ffn: Optional[bool] = False,
                 no_fusion: Optional[bool] = False):
        conv_3x3_in = ConvLayer(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True, dilation=dilation
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels, out_channels=transformer_dim,
            kernel_size=1, stride=1, use_norm=False, use_act=False
        )

        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim, out_channels=in_channels,
            kernel_size=1, stride=1, use_norm=True, use_act=True
        )
        conv_3x3_out = None
        if not no_fusion:
            conv_3x3_out = ConvLayer(
                in_channels=2 * in_channels, out_channels=in_channels,
                kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True
            )
        super(MobileViTBlock, self).__init__()
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        ffn_dims = [ffn_dim] * n_transformer_blocks

        global_rep = [
            TransformerEncoder(embed_dim=transformer_dim, ffn_latent_dim=ffn_dims[block_idx], num_heads=num_heads,
                               attn_dropout=attn_dropout, dropout=dropout, ffn_dropout=ffn_dropout)
            for block_idx in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out

        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.dilation = dilation
        self.ffn_max_dim = ffn_dims[0]
        self.ffn_min_dim = ffn_dims[-1]
        self.var_ffn = var_ffn
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, feature_map: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h
        }

        return patches, info_dict

    def folding(self, patches: Tensor, info_dict: Dict) -> Tensor:
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
        return feature_map

    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        if self.fusion is not None:
            fm = self.fusion(
                flow.cat((res, fm), dim=1)
            )
        return fm


def make_divisible(v: Union[float, int],
                   divisor: Optional[int] = 8,
                   min_value: Optional[Union[float, int]] = None) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 expand_ratio: Union[int, float],
                 dilation: int = 1
                 ) -> None:
        assert stride in [1, 2]
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(name="exp_1x1",
                             module=ConvLayer(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1,
                                              use_act=True, use_norm=True))

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(in_channels=hidden_dim, out_channels=hidden_dim, stride=stride, kernel_size=3,
                             groups=hidden_dim, use_act=True, use_norm=True, dilation=dilation)
        )

        block.add_module(name="red_1x1",
                         module=ConvLayer(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                                          use_act=False, use_norm=True))

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class GlobalPool(nn.Module):
    def __init__(self, pool_type='mean', keep_dim=False):
        """
            Global pooling
            :param pool_type: Global pool operation type (mean, rms, abs)
            :param keep_dim: Keep dimensions the same as the input or not
        """
        super(GlobalPool, self).__init__()
        pool_types = ['mean', 'rms', 'abs']
        assert pool_type in pool_types, 'Supported pool types are: {}. Got {}'.format(pool_types, pool_type)
        self.pool_type = pool_type
        self.keep_dim = keep_dim

    def _global_pool(self, x):
        assert x.dim() == 4, "Got: {}".format(x.shape)
        if self.pool_type == 'rms':
            x = x ** 2
            x = flow.mean(x, dim=[-2, -1], keepdim=self.keep_dim)
            x = x ** -0.5
        elif self.pool_type == 'abs':
            x = flow.mean(flow.abs(x), dim=[-2, -1], keepdim=self.keep_dim)
        else:
            # default is mean
            # same as AdaptiveAvgPool
            x = flow.mean(x, dim=[-2, -1], keepdim=self.keep_dim)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._global_pool(x)


class LinearLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: Optional[bool] = True
                 ) -> None:
        """
            Applies a linear transformation to the input data
            :param in_features: size of each input sample
            :param out_features:  size of each output sample
            :param bias: Add bias (learnable) or not
        """
        super(LinearLayer, self).__init__()
        self.weight = nn.Parameter(flow.Tensor(out_features, in_features))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(flow.Tensor(out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            flow.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            flow.nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.bias is not None and x.dim() == 2:
            x = flow.addmm(self.bias, x, self.weight.t())
        else:
            x = x.matmul(self.weight.t())
            if self.bias is not None:
                x += self.bias
        return x


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple or int, stride: tuple or int,
                 padding: tuple or int, dilation: int or tuple, groups: int, bias: bool, padding_mode: str
                 ):
        super(Conv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                     padding_mode=padding_mode)


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int or tuple,
                 stride: Optional[int or tuple] = 1,
                 dilation: Optional[int or tuple] = 1, groups: Optional[int] = 1,
                 bias: Optional[bool] = False, padding_mode: Optional[str] = 'zeros',
                 use_norm: Optional[bool] = True, use_act: Optional[bool] = True
                 ) -> None:
        """
            Applies a 2D convolution over an input signal composed of several input planes.
            :param opts: arguments
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param kernel_size: kernel size
            :param stride: move the kernel by this amount during convolution operation
            :param dilation: Add zeros between kernel elements to increase the effective receptive field of the kernel.
            :param groups: Number of groups. If groups=in_channels=out_channels, then it is a depth-wise convolution
            :param bias: Add bias or not
            :param padding_mode: Padding mode. Default is zeros
            :param use_norm: Use normalization layer after convolution layer or not. Default is True.
            :param use_act: Use activation layer after convolution layer/convolution layer followed by batch
            normalization or not. Default is True.
        """
        super(ConvLayer, self).__init__()

        if use_norm:
            assert not bias, 'Do not use bias when using normalization layers.'

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert isinstance(kernel_size, (tuple, list))
        assert isinstance(stride, (tuple, list))
        assert isinstance(dilation, (tuple, list))

        padding = (int((kernel_size[0] - 1) / 2) * dilation[0], int((kernel_size[1] - 1) / 2) * dilation[1])

        assert in_channels % groups == 0, \
            'Input channels are not divisible by groups. {}%{} != 0 '.format(in_channels, groups)
        assert out_channels % groups == 0, \
            'Output channels are not divisible by groups. {}%{} != 0 '.format(out_channels, groups)

        block = nn.Sequential()

        conv_layer = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                            padding_mode=padding_mode)

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            norm_layer = nn.BatchNorm2d(out_channels)
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None

        if use_act:
            act_layer = nn.SiLU()
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MobileViT(nn.Module):
    """
        MobileViT: https://arxiv.org/abs/2110.02178?context=cs.LG
    """

    def __init__(
            self,
            arch,
            num_classes=1000,
            classifier_dropout=0.1,
            pool_type='mean',
            **kwargs
    ) -> None:
        image_channels = 3
        out_channels = 16

        assert arch in CONFIG.keys()
        mobilevit_config = CONFIG[arch]

        # Segmentation architectures like Deeplab and PSPNet modifies the strides of the classification backbones
        # We allow that using `output_stride` arguments
        output_stride = kwargs.get("output_stride", None)
        dilate_l4 = dilate_l5 = False
        if output_stride == 8:
            dilate_l4 = True
            dilate_l5 = True
        elif output_stride == 16:
            dilate_l5 = True

        super(MobileViT, self).__init__()
        self.dilation = 1

        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer(
            in_channels=image_channels, out_channels=out_channels,
            kernel_size=3, stride=2, use_norm=True, use_act=True
        )

        self.model_conf_dict['conv1'] = {'in': image_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )
        self.model_conf_dict['layer1'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )
        self.model_conf_dict['layer2'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            input_channel=in_channels, cfg=mobilevit_config["layer3"]
        )
        self.model_conf_dict['layer3'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            input_channel=in_channels, cfg=mobilevit_config["layer4"], dilate=dilate_l4
        )
        self.model_conf_dict['layer4'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            input_channel=in_channels, cfg=mobilevit_config["layer5"], dilate=dilate_l5
        )
        self.model_conf_dict['layer5'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        exp_channels = min(mobilevit_config["last_layer_exp_factor"] * in_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            in_channels=in_channels, out_channels=exp_channels,
            kernel_size=1, stride=1, use_act=True, use_norm=True
        )

        self.model_conf_dict['exp_before_cls'] = {'in': in_channels, 'out': exp_channels}

        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False))
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(name="dropout", module=nn.Dropout(p=classifier_dropout, inplace=True))
        self.classifier.add_module(
            name="fc",
            module=LinearLayer(in_features=exp_channels, out_features=num_classes, bias=True)
        )

        # weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        # weight initialization
        modules = self.modules()

        for m in modules:
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Linear, LinearLayer)):
                if hasattr(m, "layer"):
                    if m.layer.weight is not None:
                        std = 0.02
                        nn.init.trunc_normal_(m.layer.weight, mean=0.0, std=std)
                    if m.layer.bias is not None:
                        nn.init.zeros_(m.layer.bias)
                else:
                    if m.weight is not None:
                        std = 0.02
                        nn.init.trunc_normal_(m.weight, mean=0.0, std=std)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _make_layer(self, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                input_channel=input_channel,
                cfg=cfg,
                dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                input_channel=input_channel,
                cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(self, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        head_dim = cfg.get("head_dim", 32)
        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        if head_dim is None:
            num_heads = cfg.get("num_heads", 4)
            if num_heads is None:
                num_heads = 4
            head_dim = transformer_dim // num_heads

        assert transformer_dim % head_dim == 0, \
            "Transformer input dimension should be divisible by head dimension. " \
            "Got {} and {}.".format(transformer_dim, head_dim)

        block.append(
            MobileViTBlock(
                in_channels=input_channel,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=cfg.get("transformer_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=0.1,
                ffn_dropout=0.0,
                attn_dropout=0.0,
                head_dim=head_dim,
                no_fusion=False,
                conv_ksize=3
            )
        )

        return nn.Sequential(*block), input_channel

    def extract_features(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.extract_features(x)
        x = self.classifier(x)
        return x


CONFIG = {
    'mobilevit_xx_small': {
        "layer1": {
            "out_channels": 16,
            "expand_ratio": 2,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2"
        },
        "layer2": {
            "out_channels": 24,
            "expand_ratio": 2,
            "num_blocks": 3,
            "stride": 2,
            "block_type": "mv2"
        },
        "layer3": {  # 28x28
            "out_channels": 48,
            "transformer_channels": 64,
            "ffn_dim": 128,
            "transformer_blocks": 2,
            "patch_h": 2,  # 8,
            "patch_w": 2,  # 8,
            "stride": 2,
            "mv_expand_ratio": 2,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer4": {  # 14x14
            "out_channels": 64,
            "transformer_channels": 80,
            "ffn_dim": 160,
            "transformer_blocks": 4,
            "patch_h": 2,  # 4,
            "patch_w": 2,  # 4,
            "stride": 2,
            "mv_expand_ratio": 2,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer5": {  # 7x7
            "out_channels": 80,
            "transformer_channels": 96,
            "ffn_dim": 192,
            "transformer_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 2,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "last_layer_exp_factor": 4
    },
    'mobilevit_x_small': {
        "layer1": {
            "out_channels": 32,
            "expand_ratio": 4,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2"
        },
        "layer2": {
            "out_channels": 48,
            "expand_ratio": 4,
            "num_blocks": 3,
            "stride": 2,
            "block_type": "mv2"
        },
        "layer3": {  # 28x28
            "out_channels": 64,
            "transformer_channels": 96,
            "ffn_dim": 192,
            "transformer_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 4,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer4": {  # 14x14
            "out_channels": 80,
            "transformer_channels": 120,
            "ffn_dim": 240,
            "transformer_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 4,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer5": {  # 7x7
            "out_channels": 96,
            "transformer_channels": 144,
            "ffn_dim": 288,
            "transformer_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 4,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "last_layer_exp_factor": 4
    },
    'mobilevit_small': {
        "layer1": {
            "out_channels": 32,
            "expand_ratio": 4,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2"
        },
        "layer2": {
            "out_channels": 64,
            "expand_ratio": 4,
            "num_blocks": 3,
            "stride": 2,
            "block_type": "mv2"
        },
        "layer3": {  # 28x28
            "out_channels": 96,
            "transformer_channels": 144,
            "ffn_dim": 288,
            "transformer_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 4,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer4": {  # 14x14
            "out_channels": 128,
            "transformer_channels": 192,
            "ffn_dim": 384,
            "transformer_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 4,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer5": {  # 7x7
            "out_channels": 160,
            "transformer_channels": 240,
            "ffn_dim": 480,
            "transformer_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 4,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "last_layer_exp_factor": 4
    }
}


def _create_mobilevit(arch: str, pretrained: bool = False, progress: bool = True, **model_kwargs):
    model = MobileViT(arch=arch, **model_kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def mobilevit_small(pretrained: bool = False, progress: bool = True, **kwargs):
    """
    Constructs MobileViT-S 224x224 model pretrained on ImageNet-1k.

    .. note::
        MobileViT-S 224x224 model from `"MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer" <https://arxiv.org/pdf/2110.02178>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mobilevit_s = flowvision.models.mobilevit_small(pretrained=False, progress=True)

    """
    return _create_mobilevit(arch='mobilevit_small', pretrained=pretrained, progress=progress, **kwargs)


@ModelCreator.register_model
def mobilevit_x_small(pretrained: bool = False, progress: bool = True, **kwargs):
    """
    Constructs MobileViT-XS 224x224 model pretrained on ImageNet-1k.

    .. note::
        MobileViT-XS 224x224 model from `"MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer" <https://arxiv.org/pdf/2110.02178>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mobilevit_xs = flowvision.models.mobilevit_x_small(pretrained=False, progress=True)

    """
    return _create_mobilevit(arch='mobilevit_x_small', pretrained=pretrained, progress=progress, **kwargs)


@ModelCreator.register_model
def mobilevit_xx_small(pretrained: bool = False, progress: bool = True, **kwargs):
    """
    Constructs MobileViT-XXS 224x224 model pretrained on ImageNet-1k.

    .. note::
        MobileViT-XXS 224x224 model from `"MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer" <https://arxiv.org/pdf/2110.02178>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> mobilevit_xxs = flowvision.models.mobilevit_xx_small(pretrained=False, progress=True)

    """
    return _create_mobilevit(arch='mobilevit_xx_small', pretrained=pretrained, progress=progress, **kwargs)
