"""
Modified from https://github.com/HRNet/HRFormer/blob/main/cls/models/hrt.py
"""

from functools import partial
import math
from typing import List, Optional, Tuple

import oneflow
from oneflow import nn, Tensor
import oneflow.nn.functional as F
from oneflow.nn.functional import linear, pad, softmax, dropout
from flowvision.layers import trunc_normal_, DropPath

from .utils import load_state_dict_from_url
from .registry import ModelCreator

# TODO: hrformer_tiny 模型转化还有问题
model_urls = {
    "hrformer_base": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/HRFormer/hrformer_base.zip",
    "hrformer_small": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/HRFormer/hrformer_small.zip",
    # "hrformer_tiny": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/HRFormer/hrformer_tiny.zip",
}


hrformer_base_cfg = {'DROP_PATH_RATE': 0.2,
 'STAGE1': {'NUM_MODULES': 1,
  'NUM_BRANCHES': 1,
  'NUM_BLOCKS': [2],
  'NUM_CHANNELS': [64],
  'NUM_HEADS': [2],
  'NUM_MLP_RATIOS': [4],
  'NUM_RESOLUTIONS': [[56, 56]],
  'BLOCK': 'BOTTLENECK'},
 'STAGE2': {'NUM_MODULES': 1,
  'NUM_BRANCHES': 2,
  'NUM_BLOCKS': [2, 2],
  'NUM_CHANNELS': [78, 156],
  'NUM_HEADS': [2, 4],
  'NUM_MLP_RATIOS': [4, 4],
  'NUM_RESOLUTIONS': [[56, 56], [28, 28]],
  'NUM_WINDOW_SIZES': [7, 7],
  'ATTN_TYPES': [[['isa_local', 'isa_local'], ['isa_local', 'isa_local']]],
  'FFN_TYPES': [[['conv_mlp', 'conv_mlp'], ['conv_mlp', 'conv_mlp']]],
  'BLOCK': 'TRANSFORMER_BLOCK'},
 'STAGE3': {'NUM_MODULES': 4,
  'NUM_BRANCHES': 3,
  'NUM_BLOCKS': [2, 2, 2],
  'NUM_CHANNELS': [78, 156, 312],
  'NUM_HEADS': [2, 4, 8],
  'NUM_MLP_RATIOS': [4, 4, 4],
  'NUM_RESOLUTIONS': [[56, 56], [28, 28], [14, 14]],
  'NUM_WINDOW_SIZES': [7, 7, 7],
  'ATTN_TYPES': [[['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']],
   [['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']],
   [['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']],
   [['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']]],
  'FFN_TYPES': [[['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp']],
   [['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp']],
   [['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp']],
   [['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp']]],
  'BLOCK': 'TRANSFORMER_BLOCK'},
 'STAGE4': {'NUM_MODULES': 2,
  'NUM_BRANCHES': 4,
  'NUM_BLOCKS': [2, 2, 2, 2],
  'NUM_CHANNELS': [78, 156, 312, 624],
  'NUM_HEADS': [2, 4, 8, 16],
  'NUM_MLP_RATIOS': [4, 4, 4, 4],
  'NUM_RESOLUTIONS': [[56, 56], [28, 28], [14, 14], [7, 7]],
  'NUM_WINDOW_SIZES': [7, 7, 7, 7],
  'ATTN_TYPES': [[['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']],
   [['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']]],
  'FFN_TYPES': [[['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp']],
   [['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp']]],
  'BLOCK': 'TRANSFORMER_BLOCK'}}

hrformer_small_cfg = {'DROP_PATH_RATE': 0.0,
 'STAGE1': {'NUM_MODULES': 1,
  'NUM_BRANCHES': 1,
  'NUM_BLOCKS': [2],
  'NUM_CHANNELS': [64],
  'NUM_HEADS': [2],
  'NUM_MLP_RATIOS': [4],
  'NUM_RESOLUTIONS': [[56, 56]],
  'BLOCK': 'BOTTLENECK'},
 'STAGE2': {'NUM_MODULES': 1,
  'NUM_BRANCHES': 2,
  'NUM_BLOCKS': [2, 2],
  'NUM_CHANNELS': [32, 64],
  'NUM_HEADS': [1, 2],
  'NUM_MLP_RATIOS': [4, 4],
  'NUM_RESOLUTIONS': [[56, 56], [28, 28]],
  'NUM_WINDOW_SIZES': [7, 7],
  'ATTN_TYPES': [[['isa_local', 'isa_local'], ['isa_local', 'isa_local']]],
  'FFN_TYPES': [[['conv_mlp', 'conv_mlp'], ['conv_mlp', 'conv_mlp']]],
  'BLOCK': 'TRANSFORMER_BLOCK'},
 'STAGE3': {'NUM_MODULES': 4,
  'NUM_BRANCHES': 3,
  'NUM_BLOCKS': [2, 2, 2],
  'NUM_CHANNELS': [32, 64, 128],
  'NUM_HEADS': [1, 2, 4],
  'NUM_MLP_RATIOS': [4, 4, 4],
  'NUM_RESOLUTIONS': [[56, 56], [28, 28], [14, 14]],
  'NUM_WINDOW_SIZES': [7, 7, 7],
  'ATTN_TYPES': [[['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']],
   [['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']],
   [['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']],
   [['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']]],
  'FFN_TYPES': [[['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp']],
   [['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp']],
   [['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp']],
   [['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp']]],
  'BLOCK': 'TRANSFORMER_BLOCK'},
 'STAGE4': {'NUM_MODULES': 2,
  'NUM_BRANCHES': 4,
  'NUM_BLOCKS': [2, 2, 2, 2],
  'NUM_CHANNELS': [32, 64, 128, 256],
  'NUM_HEADS': [1, 2, 4, 8],
  'NUM_MLP_RATIOS': [4, 4, 4, 4],
  'NUM_RESOLUTIONS': [[56, 56], [28, 28], [14, 14], [7, 7]],
  'NUM_WINDOW_SIZES': [7, 7, 7, 7],
  'ATTN_TYPES': [[['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']],
   [['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']]],
  'FFN_TYPES': [[['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp']],
   [['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp'],
    ['conv_mlp', 'conv_mlp']]],
  'BLOCK': 'TRANSFORMER_BLOCK'}}

hrformer_tiny_cfg = {'DROP_PATH_RATE': 0.0,
 'STAGE1': {'NUM_MODULES': 1,
  'NUM_BRANCHES': 1,
  'NUM_BLOCKS': [2],
  'NUM_CHANNELS': [64],
  'NUM_HEADS': [2],
  'NUM_MLP_RATIOS': [4],
  'NUM_RESOLUTIONS': [[56, 56]],
  'BLOCK': 'BOTTLENECK'},
 'STAGE2': {'NUM_MODULES': 1,
  'NUM_BRANCHES': 2,
  'NUM_BLOCKS': [2, 2],
  'NUM_CHANNELS': [18, 36],
  'NUM_HEADS': [1, 2],
  'NUM_MLP_RATIOS': [4, 4],
  'NUM_RESOLUTIONS': [[56, 56], [28, 28]],
  'NUM_WINDOW_SIZES': [7, 7],
  'ATTN_TYPES': [[['isa_local', 'isa_local'], ['isa_local', 'isa_local']]],
  'BLOCK': 'TRANSFORMER_BLOCK'},
 'STAGE3': {'NUM_MODULES': 3,
  'NUM_BRANCHES': 3,
  'NUM_BLOCKS': [2, 2, 2],
  'NUM_CHANNELS': [18, 36, 72],
  'NUM_HEADS': [1, 2, 4],
  'NUM_MLP_RATIOS': [4, 4, 4],
  'NUM_RESOLUTIONS': [[56, 56], [28, 28], [14, 14]],
  'NUM_WINDOW_SIZES': [7, 7, 7],
  'ATTN_TYPES': [[['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']],
   [['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']],
   [['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']]],
  'BLOCK': 'TRANSFORMER_BLOCK'},
 'STAGE4': {'NUM_MODULES': 2,
  'NUM_BRANCHES': 4,
  'NUM_BLOCKS': [2, 2, 2, 2],
  'NUM_CHANNELS': [18, 36, 72, 144],
  'NUM_HEADS': [1, 2, 4, 8],
  'NUM_MLP_RATIOS': [4, 4, 4, 4],
  'NUM_RESOLUTIONS': [[56, 56], [28, 28], [14, 14], [7, 7]],
  'NUM_WINDOW_SIZES': [7, 7, 7, 7],
  'ATTN_TYPES': [[['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']],
   [['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local'],
    ['isa_local', 'isa_local']]],
  'BLOCK': 'TRANSFORMER_BLOCK'}}


# helpers
def to_2tuple(x):
    return (x, x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MultiheadAttention(nn.Module):
    bias_k: Optional[Tensor]
    bias_v: Optional[Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.in_proj_bias = None
        self.in_proj_weight = None
        self.bias_k = self.bias_v = None
        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj_weight = None
        self.add_zero_attn = add_zero_attn

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        residual_attn=None,
    ):
        if not self._qkv_same_embed_dim:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                out_dim=self.vdim,
                residual_attn=residual_attn,
            )
        else:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                out_dim=self.vdim,
                residual_attn=residual_attn,
            )

    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        out_dim: Optional[Tensor] = None,
        residual_attn: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # if not torch.jit.is_scripting():
        if True:
            tens_ops = (
                query,
                key,
                value,
                in_proj_weight,
                in_proj_bias,
                bias_k,
                bias_v,
                out_proj_weight,
                out_proj_bias,
            )
            # if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(
            #     tens_ops
            # ):
            if any([type(t) is not Tensor for t in tens_ops]):
                return handle_torch_function(
                    multi_head_attention_forward,
                    tens_ops,
                    query,
                    key,
                    value,
                    embed_dim_to_check,
                    num_heads,
                    in_proj_weight,
                    in_proj_bias,
                    bias_k,
                    bias_v,
                    add_zero_attn,
                    dropout_p,
                    out_proj_weight,
                    out_proj_bias,
                    training=training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=use_separate_proj_weight,
                    q_proj_weight=q_proj_weight,
                    k_proj_weight=k_proj_weight,
                    v_proj_weight=v_proj_weight,
                    static_k=static_k,
                    static_v=static_v,
                )
        tgt_len, bsz, embed_dim = query.size()
        key = query if key is None else key
        value = query if value is None else value

        assert embed_dim == embed_dim_to_check
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        v_head_dim = out_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        q = self.q_proj(query) * scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        if attn_mask is not None:
            assert (
                attn_mask.dtype == oneflow.float32
                or attn_mask.dtype == oneflow.float64
                or attn_mask.dtype == oneflow.float16
                or attn_mask.dtype == oneflow.uint8
                or attn_mask.dtype == oneflow.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == oneflow.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(oneflow.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == oneflow.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(oneflow.bool)

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * num_heads, v_head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if add_zero_attn:
            src_len += 1
            k = oneflow.cat(
                [
                    k,
                    oneflow.zeros(
                        (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                    ),
                ],
                dim=1,
            )
            v = oneflow.cat(
                [
                    v,
                    oneflow.zeros(
                        (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                    ),
                ],
                dim=1,
            )
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        attn_output_weights = oneflow.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        """
        Attention weight for the invalid region is -inf
        """
        if attn_mask is not None:
            if attn_mask.dtype == oneflow.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        if residual_attn is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights += residual_attn.unsqueeze(0)
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        """
        Reweight the attention map before softmax().
        attn_output_weights: (b*n_head, n, hw)
        """
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output_weights = dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = oneflow.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, v_head_dim]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, out_dim)
        )
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output


class MultiheadAttentionRPE(MultiheadAttention):
    """ "Multihead Attention with extra flags on the q/k/v and out projections."""

    bias_k: Optional[oneflow.Tensor]
    bias_v: Optional[oneflow.Tensor]

    def __init__(self, *args, rpe=False, window_size=7, **kwargs):
        super(MultiheadAttentionRPE, self).__init__(*args, **kwargs)

        self.rpe = rpe
        if rpe:
            self.window_size = [window_size] * 2
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                oneflow.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                    self.num_heads,
                )
            )  # 2*Wh-1 * 2*Ww-1, nH
            # get pair-wise relative position index for each token inside the window
            coords_h = oneflow.arange(self.window_size[0])
            coords_w = oneflow.arange(self.window_size[1])
            # coords = oneflow.stack(oneflow.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords = oneflow.stack(oneflow.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
            coords_flatten = oneflow.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        do_qkv_proj=True,
        do_out_proj=True,
        rpe=True,
    ):
        if not self._qkv_same_embed_dim:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                out_dim=self.vdim,
                do_qkv_proj=do_qkv_proj,
                do_out_proj=do_out_proj,
                rpe=rpe,
            )
        else:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                out_dim=self.vdim,
                do_qkv_proj=do_qkv_proj,
                do_out_proj=do_out_proj,
                rpe=rpe,
            )

    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        out_dim: Optional[Tensor] = None,
        do_qkv_proj: bool = True,
        do_out_proj: bool = True,
        rpe=True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # if not torch.jit.is_scripting():
        if True:
            tens_ops = (
                query,
                key,
                value,
                in_proj_weight,
                in_proj_bias,
                bias_k,
                bias_v,
                out_proj_weight,
                out_proj_bias,
            )

            # 原来的if，oneflow 没有handle_torch_function 和 has_torch_function
            # if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(
            #     tens_ops
            # ):
            #     return handle_torch_function(
            #         multi_head_attention_forward,
            #         tens_ops,
            #         query,
            #         key,
            #         value,
            #         embed_dim_to_check,
            #         num_heads,
            #         in_proj_weight,
            #         in_proj_bias,
            #         bias_k,
            #         bias_v,
            #         add_zero_attn,
            #         dropout_p,
            #         out_proj_weight,
            #         out_proj_bias,
            #         training=training,
            #         key_padding_mask=key_padding_mask,
            #         need_weights=need_weights,
            #         attn_mask=attn_mask,
            #         use_separate_proj_weight=use_separate_proj_weight,
            #         q_proj_weight=q_proj_weight,
            #         k_proj_weight=k_proj_weight,
            #         v_proj_weight=v_proj_weight,
            #         static_k=static_k,
            #         static_v=static_v,
            #     )

            # 修改后
            # if any([type(t) is not Tensor for t in tens_ops]):
            #     return multi_head_attention_forward,
            #         tens_ops,
            #         query,
            #         key,
            #         value,
            #         embed_dim_to_check,
            #         num_heads,
            #         in_proj_weight,
            #         in_proj_bias,
            #         bias_k,
            #         bias_v,
            #         add_zero_attn,
            #         dropout_p,
            #         out_proj_weight,
            #         out_proj_bias,
            #         training=training,
            #         key_padding_mask=key_padding_mask,
            #         need_weights=need_weights,
            #         attn_mask=attn_mask,
            #         use_separate_proj_weight=use_separate_proj_weight,
            #         q_proj_weight=q_proj_weight,
            #         k_proj_weight=k_proj_weight,
            #         v_proj_weight=v_proj_weight,
            #         static_k=static_k,
            #         static_v=static_v,

        tgt_len, bsz, embed_dim = query.size()
        key = query if key is None else key
        value = query if value is None else value

        assert embed_dim == embed_dim_to_check
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        v_head_dim = out_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        # whether or not use the original query/key/value
        q = self.q_proj(query) * scaling if do_qkv_proj else query
        k = self.k_proj(key) if do_qkv_proj else key
        v = self.v_proj(value) if do_qkv_proj else value

        if attn_mask is not None:
            assert (
                attn_mask.dtype == oneflow.float32
                or attn_mask.dtype == oneflow.float64
                or attn_mask.dtype == oneflow.float16
                or attn_mask.dtype == oneflow.uint8
                or attn_mask.dtype == oneflow.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == oneflow.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(oneflow.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == oneflow.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(oneflow.bool)

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * num_heads, v_head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if add_zero_attn:
            src_len += 1
            k = oneflow.cat(
                [
                    k,
                    oneflow.zeros(
                        (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                    ),
                ],
                dim=1,
            )
            v = oneflow.cat(
                [
                    v,
                    oneflow.zeros(
                        (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                    ),
                ],
                dim=1,
            )
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        attn_output_weights = oneflow.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        """
        Add relative position embedding
        """
        if self.rpe and rpe:
            # NOTE: for simplicity, we assume the src_len == tgt_len == window_size**2 here
            assert (
                src_len == self.window_size[0] * self.window_size[1]
                and tgt_len == self.window_size[0] * self.window_size[1]
            ), f"src{src_len}, tgt{tgt_len}, window{self.window_size[0]}"
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            ) + relative_position_bias.unsqueeze(0)
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        """
        Attention weight for the invalid region is -inf
        """
        if attn_mask is not None:
            if attn_mask.dtype == oneflow.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        """
        Reweight the attention map before softmax().
        attn_output_weights: (b*n_head, n, hw)
        """
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output_weights = dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = oneflow.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, v_head_dim]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, out_dim)
        )
        if do_out_proj:
            attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, q, k, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, q, k  # additionaly return the query and key


class PadBlock(object):
    """ "Make the size of feature map divisible by local group size."""

    def __init__(self, local_group_size=7):
        self.lgs = local_group_size
        if not isinstance(self.lgs, (tuple, list)):
            self.lgs = to_2tuple(self.lgs)
        assert len(self.lgs) == 2

    def pad_if_needed(self, x, size):
        n, h, w, c = size
        pad_h = math.ceil(h / self.lgs[0]) * self.lgs[0] - h
        pad_w = math.ceil(w / self.lgs[1]) * self.lgs[1] - w
        if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
            return F.pad(
                x,
                (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
        return x

    def depad_if_needed(self, x, size):
        n, h, w, c = size
        pad_h = math.ceil(h / self.lgs[0]) * self.lgs[0] - h
        pad_w = math.ceil(w / self.lgs[1]) * self.lgs[1] - w
        if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
            return x[:, pad_h // 2 : pad_h // 2 + h, pad_w // 2 : pad_w // 2 + w, :]
        return x


class LocalPermuteModule(object):
    """ "Permute the feature map to gather pixels in local groups, and the reverse permutation"""

    def __init__(self, local_group_size=7):
        self.lgs = local_group_size
        if not isinstance(self.lgs, (tuple, list)):
            self.lgs = to_2tuple(self.lgs)
        assert len(self.lgs) == 2
# TODO: Add OneFlow backend into einops
    def permute(self, x, size):
        n, h, w, c = size
        qh = h // self.lgs[0]
        ph = self.lgs[0]
        qw = w // self.lgs[0]
        pw = self.lgs[0]
        x = x.view(n, qh, ph, qw, pw, c)
        x_rearrange = oneflow.permute(x, (2, 4, 0, 1, 3, 5)) 
        return x_rearrange.view(ph*pw, n*qh*qw, c)

    #TODO: use einops.rearrange replace
    def rev_permute(self, x, size):
        n, h, w, c = size
        ph = self.lgs[0]
        pw = self.lgs[0]
        qh = h // self.lgs[0]
        qw = w // self.lgs[0]
        x = x.view(ph, pw, n, qh, qw, c)
        x_rearrange = oneflow.permute(x, (2, 3, 0, 4, 1, 5))
        return x_rearrange.view(n, qh*ph, qw*pw, c)


class MultiheadISAAttention(nn.Module):
    r"""interlaced sparse multi-head self attention (ISA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): Window size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        window_size=7,
        attn_type="isa_local",
        rpe=True,
        **kwargs,
    ):
        super(MultiheadISAAttention, self).__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn_type = attn_type
        self.with_rpe = rpe

        self.attn = MultiheadAttentionRPE(
            embed_dim, num_heads, rpe=rpe, window_size=window_size, **kwargs
        )
        self.pad_helper = PadBlock(window_size)
        assert attn_type in ["isa_local"]
        if attn_type == "isa_local":
            self.permute_helper = LocalPermuteModule(window_size)
        else:
            raise NotImplementedError("We only support ['isa_local'] Now.")

    def forward(self, x, H, W, **kwargs):
        # H, W = self.input_resolution
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        # attention
        if self.attn_type in ["isa_local"]:
            # pad
            x_pad = self.pad_helper.pad_if_needed(x, x.size())
            # permute
            x_permute = self.permute_helper.permute(x_pad, x_pad.size())
            # attention
            out, _, _ = self.attn(
                x_permute, x_permute, x_permute, rpe=self.with_rpe, **kwargs
            )
            # reverse permutation
            out = self.permute_helper.rev_permute(out, x_pad.size())
        else:
            raise NotImplementedError("We only support ['isa_local'] Now.")
        # de-pad, pooling with `ceil_mode=True` will do implicit padding, so we need to remove it, too
        out = self.pad_helper.depad_if_needed(out, x.size())
        return out.reshape(B, N, C)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class BottleneckDWP(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckDWP, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=planes,
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MlpDWBN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        dw_act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = act_layer()
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.dw3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1,
        )
        self.act2 = dw_act_layer()
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = act_layer()
        self.norm3 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        if len(x.shape) == 3:
            B, N, C = x.shape
            if N == (H * W + 1):
                cls_tokens = x[:, 0, :]
                x_ = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)
            else:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)

            x_ = self.fc1(x_)
            x_ = self.norm1(x_)
            x_ = self.act1(x_)
            x_ = self.dw3x3(x_)
            x_ = self.norm2(x_)
            x_ = self.act2(x_)
            x_ = self.drop(x_)
            x_ = self.fc2(x_)
            x_ = self.norm3(x_)
            x_ = self.act3(x_)
            x_ = self.drop(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            if N == (H * W + 1):
                x = oneflow.cat((cls_tokens.unsqueeze(1), x_), dim=1)
            else:
                x = x_
            return x

        elif len(x.shape) == 4:
            x = self.fc1(x)
            x = self.norm1(x)
            x = self.act1(x)
            x = self.dw3x3(x)
            x = self.norm2(x)
            x = self.act2(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.norm3(x)
            x = self.act3(x)
            x = self.drop(x)
            return x

        else:
            raise RuntimeError("Unsupported input shape: {}".format(x.shape))


class GeneralTransformerBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_type="isa_local",
        ffn_type="conv_mlp",
    ):
        super().__init__()
        self.dim = inplanes
        self.out_dim = planes
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn_type = attn_type
        self.ffn_type = ffn_type
        self.mlp_ratio = mlp_ratio

        if self.attn_type in ["conv"]:
            """modified basic block with seperable 3x3 convolution"""
            self.sep_conv1 = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=inplanes,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
                nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            )
            self.sep_conv2 = nn.Sequential(
                nn.Conv2d(
                    planes,
                    planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=planes,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
                nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
            )
            self.relu = nn.ReLU(inplace=True)
        elif self.attn_type in ["isa_local"]:
            self.attn = MultiheadISAAttention(
                self.dim,
                num_heads=num_heads,
                window_size=window_size,
                attn_type=attn_type,
                rpe=True,
                dropout=attn_drop,
            )
            self.norm1 = norm_layer(self.dim)
            self.norm2 = norm_layer(self.out_dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            mlp_hidden_dim = int(self.dim * mlp_ratio)

            if self.ffn_type in ["conv_mlp"]:
                self.mlp = MlpDWBN(
                    in_features=self.dim,
                    hidden_features=mlp_hidden_dim,
                    out_features=self.out_dim,
                    act_layer=act_layer,
                    drop=drop,
                )
            elif self.ffn_type in ["identity"]:
                self.mlp = nn.Identity()
            else:
                raise RuntimeError("Unsupported ffn type: {}".format(self.ffn_type))

        else:
            raise RuntimeError("Unsupported attention type: {}".format(self.attn_type))

    def forward(self, x):
        if self.attn_type in ["conv"]:
            residual = x
            out = self.sep_conv1(x)
            out = self.sep_conv2(out)
            out += residual
            out = self.relu(out)
            return out
        elif self.attn_type in ["isa_local"]:
            B, C, H, W = x.size()
            # reshape
            x = x.view(B, C, -1).permute(0, 2, 1)
            # Attention
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            # FFN
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            # reshape
            x = x.permute(0, 2, 1).view(B, C, H, W)
            return x
        else:
            B, C, H, W = x.size()
            # reshape
            x = x.view(B, C, -1).permute(0, 2, 1)
            # Attention
            x = x + self.drop_path(self.attn(self.norm1(x)))
            # FFN
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            # reshape
            x = x.permute(0, 2, 1).view(B, C, H, W)
            return x


blocks_dict = {
    "BOTTLENECK": Bottleneck,
    "TRANSFORMER_BLOCK": GeneralTransformerBlock,
}

BN_MOMENTUM = 0.1


class HighResolutionTransformerModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        num_input_resolutions,
        attn_types,
        ffn_types,
        multi_scale_output=True,
        drop_paths=0.0,
    ):
        """
        Args:
            num_heads: the number of head witin each MHSA
            num_window_sizes: the window size for the local self-attention
            num_input_resolutions: the spatial height/width of the input feature maps.
        """
        super(HighResolutionTransformerModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            num_input_resolutions,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            attn_types,
            ffn_types,
            drop_paths,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

        self.num_heads = num_heads
        self.num_window_sizes = num_window_sizes
        self.num_mlp_ratios = num_mlp_ratios
        self.num_input_resolutions = num_input_resolutions
        self.attn_types = attn_types
        self.ffn_types = ffn_types

    def _check_branches(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            print(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(
        self,
        branch_index,
        block,
        num_blocks,
        num_channels,
        num_input_resolutions,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        attn_types,
        ffn_types,
        drop_paths,
        stride=1,
    ):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                input_resolution=num_input_resolutions[branch_index],
                num_heads=num_heads[branch_index],
                window_size=num_window_sizes[branch_index],
                mlp_ratio=num_mlp_ratios[branch_index],
                attn_type=attn_types[branch_index][0],
                ffn_type=ffn_types[branch_index][0],
                drop_path=drop_paths[0],
            )
        )

        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    input_resolution=num_input_resolutions[branch_index],
                    num_heads=num_heads[branch_index],
                    window_size=num_window_sizes[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    attn_type=attn_types[branch_index][i],
                    ffn_type=ffn_types[branch_index][i],
                    drop_path=drop_paths[i],
                )
            )
        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches,
        block,
        num_blocks,
        num_channels,
        num_input_resolutions,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        attn_types,
        ffn_types,
        drop_paths,
    ):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    num_input_resolutions,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    attn_types,
                    ffn_types,
                    drop_paths,
                )
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                kernel_size=1,
                                stride=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=num_inchannels[j],
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_inchannels[j], momentum=BN_MOMENTUM
                                    ),
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=num_inchannels[j],
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_inchannels[j], momentum=BN_MOMENTUM
                                    ),
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                    nn.ReLU(False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HighResolutionTransformer(nn.Module):
    def __init__(self, cfg, num_classes=1000, **kwargs):
        super(HighResolutionTransformer, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # stochastic depth
        depth_s2 = cfg["STAGE2"]["NUM_BLOCKS"][0] * cfg["STAGE2"]["NUM_MODULES"]
        depth_s3 = cfg["STAGE3"]["NUM_BLOCKS"][0] * cfg["STAGE3"]["NUM_MODULES"]
        depth_s4 = cfg["STAGE4"]["NUM_BLOCKS"][0] * cfg["STAGE4"]["NUM_MODULES"]
        depths = [depth_s2, depth_s3, depth_s4]
        drop_path_rate = cfg["DROP_PATH_RATE"]
        dpr = [x.item() for x in oneflow.linspace(0, drop_path_rate, sum(depths))]

        self.stage1_cfg = cfg["STAGE1"]
        num_channels = self.stage1_cfg["NUM_CHANNELS"][0]
        block = blocks_dict[self.stage1_cfg["BLOCK"]]
        num_blocks = self.stage1_cfg["NUM_BLOCKS"][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = cfg["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels
        )
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, drop_paths=dpr[0:depth_s2]
        )

        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg,
            num_channels,
            drop_paths=dpr[depth_s2 : depth_s2 + depth_s3],
        )

        self.stage4_cfg = cfg["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multi_scale_output=True,
            drop_paths=dpr[depth_s2 + depth_s3 :],
        )

        # Classification Head
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(
            pre_stage_channels
        )

        self.classifier = nn.Linear(2048, num_classes)

    def _make_head(self, pre_stage_channels):
        head_block = BottleneckDWP
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(
                head_block, channels, head_channels[i], 1, stride=1
            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_channels,
                ),
                nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_channels_cur_layer[i], momentum=BN_MOMENTUM
                            ),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(
        self,
        block,
        inplanes,
        planes,
        blocks,
        input_resolution=None,
        num_heads=1,
        stride=1,
        window_size=7,
        halo_size=1,
        mlp_ratio=4.0,
        q_dilation=1,
        kv_dilation=1,
        sr_ratio=1,
        attn_type="msw",
    ):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []

        if isinstance(block, GeneralTransformerBlock):
            layers.append(
                block(
                    inplanes,
                    planes,
                    num_heads,
                    window_size,
                    halo_size,
                    mlp_ratio,
                    q_dilation,
                    kv_dilation,
                    sr_ratio,
                    attn_type,
                )
            )
        else:
            layers.append(block(inplanes, planes, stride, downsample))

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(
        self, layer_config, num_inchannels, multi_scale_output=True, drop_paths=0.0
    ):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        num_heads = layer_config["NUM_HEADS"]
        num_window_sizes = layer_config["NUM_WINDOW_SIZES"]
        num_mlp_ratios = layer_config["NUM_MLP_RATIOS"]
        num_input_resolutions = layer_config["NUM_RESOLUTIONS"]
        attn_types = layer_config["ATTN_TYPES"]
        ffn_types = layer_config["FFN_TYPES"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionTransformerModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    num_input_resolutions,
                    attn_types[i],
                    ffn_types[i],
                    reset_multi_scale_output,
                    drop_paths=drop_paths[num_blocks[0] * i : num_blocks[0] * (i + 1)],
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)

        y = self.final_layer(y)
        y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        y = self.classifier(y)

        return y

    def init_weights(
        self,
        pretrained="",
    ):
        logger.info("=> init weights from normal distribution")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = oneflow.load(pretrained)
            logger.info("=> loading pretrained model {}".format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
            }
            for k, _ in pretrained_dict.items():
                logger.info("=> loading {} pretrained model {}".format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def _create_hrformer(arch, pretrained=False, progress=True, **model_kwargs):
    model = HighResolutionTransformer(**model_kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def hrformer_base(pretrained=False, progress=True, **kwargs):
    """
    Constructs the HRFormer-Base model trained on ImageNet2012.

    .. note::
        HRFormer-Base model from `HRFormer: High-Resolution Transformer for Dense Prediction <https://arxiv.org/abs/2110.09408>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> hrformer_base = flowvision.models.hrformer_base(pretrained=False, progress=True)

    """
    model_kwargs = dict(cfg=hrformer_base_cfg)
    return _create_hrformer(
        "hrformer_base", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def hrformer_small(pretrained=False, progress=True, **kwargs):
    """
    Constructs the HRFormer-Small model trained on ImageNet2012.

    .. note::
        HRFormer-Small model from `HRFormer: High-Resolution Transformer for Dense Prediction <https://arxiv.org/abs/2110.09408>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> hrformer_small = flowvision.models.hrformer_small(pretrained=False, progress=True)

    """
    model_kwargs = dict(cfg=hrformer_small_cfg)
    return _create_hrformer(
        "hrformer_small", pretrained=pretrained, progress=progress, **model_kwargs
    )


@ModelCreator.register_model
def hrformer_tiny(pretrained=False, progress=True, **kwargs):
    """
    Constructs the HRFormer-Tiny model trained on ImageNet2012.

    .. note::
        HRFormer-Tiny model from `HRFormer: High-Resolution Transformer for Dense Prediction <https://arxiv.org/abs/2110.09408>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> hrformer_tiny = flowvision.models.hrformer_tiny(pretrained=False, progress=True)

    """
    model_kwargs = dict(cfg=hrformer_tiny_cfg)
    return _create_hrformer(
        "hrformer_tiny", pretrained=pretrained, progress=progress, **model_kwargs
    )
