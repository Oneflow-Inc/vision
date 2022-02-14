"""
Modified from https://github.com/IBM/RegionViT
"""
import math
import copy
from functools import partial

import numpy as np
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

from ..layers import LayerNorm2d, DropPath, Mlp, trunc_normal_
from .helpers import to_2tuple
from .utils import load_state_dict_from_url
from .registry import ModelCreator


model_urls = {
    "tiny_w14": "",
    "tiny": "",
    "small": "",
    "small_w14": "",
    "small_w14_peg": "",
    "medium": "",
    "base": "",
    "base_w14": "",
    "base_w14_peg": "",
}


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, patch_conv_type='linear'):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches


        if patch_conv_type == '3conv':
            if patch_size[0] == 4:
                tmp = [
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=3, stride=2, padding=1),
                    LayerNorm2d(embed_dim // 4),
                    nn.GELU(),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    LayerNorm2d(embed_dim // 2),
                    nn.GELU(),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                ]
            else:
                raise ValueError(f"Unknown patch size {patch_size[0]}")
            self.proj = nn.Sequential(*tmp)
        else:
            if patch_conv_type == '1conv':
                kernel_size = (2 * patch_size[0], 2 * patch_size[1])
                stride = (patch_size[0], patch_size[1])
                padding = (patch_size[0] - 1, patch_size[1] - 1)
            else:
                kernel_size = patch_size
                stride = patch_size
                padding = 0

            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                                  stride=stride, padding=padding)

    def forward(self, x, extra_padding=False):
        B, C, H, W = x.shape
        if extra_padding and (H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0):
            p_l = (self.patch_size[1] - W % self.patch_size[1]) // 2
            p_r = (self.patch_size[1] - W % self.patch_size[1]) - p_l
            p_t = (self.patch_size[0] - H % self.patch_size[0]) // 2
            p_b = (self.patch_size[0] - H % self.patch_size[0]) - p_t
            x = F.pad(x, (p_l, p_r, p_t, p_b))
        x = self.proj(x)
        return x


class AttentionWithRelPos(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 attn_map_dim=None, num_cls_tokens=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.num_cls_tokens = num_cls_tokens
        if attn_map_dim is not None:
            one_dim = attn_map_dim[0]
            rel_pos_dim = (2 * one_dim - 1)
            self.rel_pos = nn.Parameter(flow.zeros(num_heads, rel_pos_dim ** 2))
            tmp = flow.arange(rel_pos_dim ** 2).reshape((rel_pos_dim, rel_pos_dim))
            out = []
            offset_x = offset_y = one_dim // 2
            for y in range(one_dim):
                for x in range(one_dim):
                    for dy in range(one_dim):
                        for dx in range(one_dim):
                            out.append(tmp[dy - y + offset_y, dx - x + offset_x].numpy())
            # self.rel_pos_index = flow.tensor(out)
            self.rel_pos_index = flow.tensor(out, dtype=flow.long)
            trunc_normal_(self.rel_pos, std=.02)
        else:
            self.rel_pos = None

    def forward(self, x, patch_attn=False, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = flow.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.rel_pos is not None and patch_attn:
            # use for the indicating patch + cls:
            rel_pos = self.rel_pos[:, self.rel_pos_index.to(attn.device)].reshape(self.num_heads, N - self.num_cls_tokens, N - self.num_cls_tokens)
            attn[:, :, self.num_cls_tokens:, self.num_cls_tokens:] = attn[:, :, self.num_cls_tokens:, self.num_cls_tokens:] + rel_pos

        if mask is not None:
            ## mask is only (BH_sW_s)(ksks)(ksks), need to expand it
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn = attn.masked_fill(mask == 0, flow.tensor(np.finfo(attn.dtype).min))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = flow.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class R2LAttentionPlusFFN(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop_path=0., attn_drop=0., drop=0.,
                 cls_attn=True):
        super().__init__()

        if not isinstance(kernel_size, (tuple, list)):
            kernel_size = [(kernel_size, kernel_size), (kernel_size, kernel_size), 0]
        self.kernel_size = kernel_size

        if cls_attn:
            self.norm0 = norm_layer(input_channels)
        else:
            self.norm0 = None

        self.norm1 = norm_layer(input_channels)
        self.attn = AttentionWithRelPos(
            input_channels, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            attn_map_dim=(kernel_size[0][0], kernel_size[0][1]), num_cls_tokens=1)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(input_channels)
        self.mlp = Mlp(in_features=input_channels, hidden_features=int(output_channels * mlp_ratio), out_features=output_channels, act_layer=act_layer, drop=drop)

        self.expand = nn.Sequential(
            norm_layer(input_channels),
            act_layer(),
            nn.Linear(input_channels, output_channels)
        ) if input_channels != output_channels else None
        self.output_channels = output_channels
        self.input_channels = input_channels

    def forward(self, xs):
        out, B, H, W, mask = xs
        cls_tokens = out[:, 0:1, ...]

        C = cls_tokens.shape[-1]
        cls_tokens = cls_tokens.reshape(B, -1, C)  # (N)x(H/sxW/s)xC

        if self.norm0 is not None:
            cls_tokens = cls_tokens + self.drop_path(self.attn(self.norm0(cls_tokens)))  # (N)x(H/sxK/s)xC

        # ks, stride, padding = self.kernel_size
        cls_tokens = cls_tokens.reshape(-1, 1, C)  # (NxH/sxK/s)x1xC

        out = flow.cat((cls_tokens, out[:, 1:, ...]), dim=1)
        tmp = out

        tmp = tmp + self.drop_path(self.attn(self.norm1(tmp), patch_attn=True, mask=mask))
        identity = self.expand(tmp) if self.expand is not None else tmp
        tmp = identity + self.drop_path(self.mlp(self.norm2(tmp)))

        return tmp


class Projection(nn.Module):
    def __init__(self, input_channels, output_channels, act_layer, mode='sc'):
        super().__init__()
        tmp = []
        if 'c' in mode:
            ks = 2 if 's' in mode else 1
            if ks == 2:
                stride = ks
                ks = ks + 1
                padding = ks // 2
            else:
                stride = ks
                padding = 0

            if input_channels == output_channels and ks == 1:
                tmp.append(nn.Identity())
            else:
                tmp.extend([
                    LayerNorm2d(input_channels),
                    act_layer(),
                ])
                tmp.append(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=ks, stride=stride, padding=padding, groups=input_channels))

        self.proj = nn.Sequential(*tmp)
        self.proj_cls = self.proj

    def forward(self, xs):
        cls_tokens, patch_tokens = xs
        # x: BxCxHxW
        cls_tokens = self.proj_cls(cls_tokens)
        patch_tokens = self.proj(patch_tokens)
        return cls_tokens, patch_tokens


def convert_to_flatten_layout(cls_tokens, patch_tokens, ws):
    """
    Convert the token layer in a flatten form, it will speed up the model.

    Furthermore, it also handle the case that if the size between regional tokens and local tokens are not consistent.
    """
    # padding if needed, and all paddings are happened at bottom and right.
    B, C, H, W = patch_tokens.shape
    _, _, H_ks, W_ks = cls_tokens.shape
    need_mask = False
    p_l, p_r, p_t, p_b = 0, 0, 0, 0
    if H % (H_ks * ws) != 0 or W % (W_ks * ws) != 0:
        p_l, p_r = 0, W_ks * ws - W
        p_t, p_b = 0, H_ks * ws - H
        patch_tokens = F.pad(patch_tokens, (p_l, p_r, p_t, p_b))
        need_mask = True

    B, C, H, W = patch_tokens.shape
    kernel_size = (H // H_ks, W // W_ks)
    tmp = nn.Unfold(kernel_size=kernel_size, stride=kernel_size, padding=(0, 0))(patch_tokens)  # Nx(Cxksxks)x(H/sxK/s)
    patch_tokens = tmp.transpose(1, 2).reshape(-1, C, kernel_size[0] * kernel_size[1]).transpose(-2, -1)  # (NxH/sxK/s)x(ksxks)xC

    if need_mask:
        BH_sK_s, ksks, C = patch_tokens.shape
        H_s, W_s = H // ws, W // ws
        mask = flow.ones(BH_sK_s // B, 1 + ksks, 1 + ksks, device=patch_tokens.device, dtype=flow.float)
        right = flow.zeros(1 + ksks, 1 + ksks, device=patch_tokens.device, dtype=flow.float)
        tmp = flow.zeros(ws, ws, device=patch_tokens.device, dtype=flow.float)
        tmp[0:(ws - p_r), 0:(ws - p_r)] = 1.
        tmp = tmp.repeat(ws, ws)
        right[1:, 1:] = tmp
        right[0, 0] = 1
        right[0, 1:] = flow.tensor([1.] * (ws - p_r) + [0.] * p_r).repeat(ws).to(right.device)
        right[1:, 0] = flow.tensor([1.] * (ws - p_r) + [0.] * p_r).repeat(ws).to(right.device)
        bottom = flow.zeros_like(right)
        bottom[0:ws * (ws - p_b) + 1, 0:ws * (ws - p_b) + 1] = 1.
        bottom_right = copy.deepcopy(right)
        bottom_right[0:ws * (ws - p_b) + 1, 0:ws * (ws - p_b) + 1] = 1.

        mask[W_s - 1:(H_s - 1) * W_s:W_s, ...] = right
        mask[(H_s - 1) * W_s:, ...] = bottom
        mask[-1, ...] = bottom_right
        mask = mask.repeat(B, 1, 1)
    else:
        mask = None

    cls_tokens = cls_tokens.flatten(2).transpose(-2, -1)  # (N)x(H/sxK/s)xC
    cls_tokens = cls_tokens.reshape(-1, 1, cls_tokens.size(-1))  # (NxH/sxK/s)x1xC

    out = flow.cat((cls_tokens, patch_tokens), dim=1)

    return out, mask, p_l, p_r, p_t, p_b, B, C, H, W


def convert_to_spatial_layout(out, output_channels, B, H, W, kernel_size, mask, p_l, p_r, p_t, p_b):
    """
    Convert the token layer from flatten into 2-D, will be used to downsample the spatial dimension.
    """
    cls_tokens = out[:, 0:1, ...]
    patch_tokens = out[:, 1:, ...]
    # cls_tokens: (BxH/sxW/s)x(1)xC, patch_tokens: (BxH/sxW/s)x(ksxks)xC
    C = output_channels
    kernel_size = kernel_size[0]
    H_ks = H // kernel_size[0]
    W_ks = W // kernel_size[1]
    # reorganize data, need to convert back to cls_tokens: BxCxH/sxW/s, patch_tokens: BxCxHxW
    cls_tokens = cls_tokens.reshape(B, -1, C).transpose(-2, -1).reshape(B, C, H_ks, W_ks)
    patch_tokens = patch_tokens.transpose(1, 2).reshape((B, -1, kernel_size[0] * kernel_size[1] * C)).transpose(1, 2)
    patch_tokens = nn.Fold(output_size=(H, W), kernel_size=kernel_size, stride=kernel_size, padding=(0, 0))(patch_tokens)

    if mask is not None:
        if p_b > 0:
            patch_tokens = patch_tokens[:, :, :-p_b, :]
        if p_r > 0:
            patch_tokens = patch_tokens[:, :, :, :-p_r]

    return cls_tokens, patch_tokens


class ConvAttBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, num_blocks, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, pool='sc',
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop_path_rate=(0.,), attn_drop_rate=0., drop_rate=0.,
                 cls_attn=True, peg=False):
        super().__init__()
        tmp = []
        if pool:
            tmp.append(Projection(input_channels, output_channels, act_layer=act_layer, mode=pool))

        for i in range(num_blocks):
            kernel_size_ = kernel_size
            tmp.append(R2LAttentionPlusFFN(output_channels, output_channels, kernel_size_, num_heads, mlp_ratio, qkv_bias, qk_scale,
                                           act_layer=act_layer, norm_layer=norm_layer, drop_path=drop_path_rate[i], attn_drop=attn_drop_rate, drop=drop_rate,
                                           cls_attn=cls_attn))

        self.block = nn.ModuleList(tmp)
        self.output_channels = output_channels
        self.ws = kernel_size
        if not isinstance(kernel_size, (tuple, list)):
            kernel_size = [(kernel_size, kernel_size), (kernel_size, kernel_size), 0]
        self.kernel_size = kernel_size

        self.peg = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, groups=output_channels, bias=False) if peg else None

    def forward(self, xs):
        cls_tokens, patch_tokens = xs
        cls_tokens, patch_tokens = self.block[0]((cls_tokens, patch_tokens))
        out, mask, p_l, p_r, p_t, p_b, B, C, H, W = convert_to_flatten_layout(cls_tokens, patch_tokens, self.ws)
        for i in range(1, len(self.block)):
            blk = self.block[i]

            out = blk((out, B, H, W, mask))
            if self.peg is not None and i == 1:
                cls_tokens, patch_tokens = convert_to_spatial_layout(out, self.output_channels, B, H, W, self.kernel_size, mask, p_l, p_r, p_t, p_b)
                cls_tokens = cls_tokens + self.peg(cls_tokens)
                patch_tokens = patch_tokens + self.peg(patch_tokens)
                out, mask, p_l, p_r, p_t, p_b, B, C, H, W = convert_to_flatten_layout(cls_tokens, patch_tokens, self.ws)

        cls_tokens, patch_tokens = convert_to_spatial_layout(out, self.output_channels, B, H, W, self.kernel_size, mask, p_l, p_r, p_t, p_b)
        return cls_tokens, patch_tokens


class RegionViT(nn.Module):
    """
    Note:
        The variable naming mapping between codes and papers:
        - cls_tokens -> regional tokens
        - patch_tokens -> local tokens
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=(768,), depth=(12,),
                 num_heads=(12,), mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 # regionvit parameters
                 kernel_sizes=None, downsampling=None,
                 patch_conv_type='3conv',
                 computed_cls_token=True, peg=False,
                 det_norm=False):

        super().__init__()
        self.num_classes = num_classes
        self.kernel_sizes = kernel_sizes
        self.num_features = embed_dim[-1]  # num_features for consistency with other models
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0],
            patch_conv_type=patch_conv_type)

        if not isinstance(mlp_ratio, (list, tuple)):
            mlp_ratio = [mlp_ratio] * len(depth)

        self.computed_cls_token = computed_cls_token
        self.cls_token = PatchEmbed(
            img_size=img_size, patch_size=patch_size * kernel_sizes[0], in_chans=in_chans, embed_dim=embed_dim[0],
            patch_conv_type='linear'
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        total_depth = sum(depth)
        dpr = [x.item() for x in flow.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.layers = nn.ModuleList()
        for i in range(len(embed_dim) - 1):
            curr_depth = depth[i]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            
            self.layers.append(
                ConvAttBlock(embed_dim[i], embed_dim[i + 1], kernel_size=kernel_sizes[i], num_blocks=depth[i], drop_path_rate=dpr_,
                             num_heads=num_heads[i], mlp_ratio=mlp_ratio[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                             pool=downsampling[i], norm_layer=norm_layer, attn_drop_rate=attn_drop_rate, drop_rate=drop_rate,
                             cls_attn=True, peg=peg)
            )
            dpr_ptr += curr_depth
        self.norm = norm_layer(embed_dim[-1])

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        if not computed_cls_token:
            trunc_normal_(self.cls_token, std=.02)

        self.det_norm = det_norm
        if self.det_norm:
            # add a norm layer for the outputs at each stage, for detection
            for i in range(4):
                layer = LayerNorm2d(embed_dim[1 + i])
                layer_name = f'norm{i}'
                self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        if not self.computed_cls_token:
            return {'cls_token'}
        else:
            return {}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, detection=False):
        o_x = x
        x = self.patch_embed(x)
        # B x branches x classes
        cls_tokens = self.cls_token(o_x, extra_padding=True)
        x = self.pos_drop(x)  # N C H W
        tmp_out = []

        for idx, layer in enumerate(self.layers):
            cls_tokens, x = layer((cls_tokens, x))
            if self.det_norm:
                norm_layer = getattr(self, f'norm{idx}')
                x = norm_layer(x)
            tmp_out.append(x)

        if detection:
            return tmp_out

        N, C, H, W = cls_tokens.shape
        cls_tokens = cls_tokens.reshape(N, C, -1).transpose(1, 2)
        cls_tokens = self.norm(cls_tokens)
        out = flow.mean(cls_tokens, dim=1)

        return out

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


_model_cfg = {
    'tiny': {
        'img_size': 224,
        'patch_conv_type': '3conv',
        'patch_size': 4,
        'embed_dim': [64, 64, 128, 256, 512],
        'num_heads': [2, 4, 8, 16],
        'mlp_ratio': 4.,
        'depth': [2, 2, 8, 2],
        'kernel_sizes': [7, 7, 7, 7],  # 8x8, 4x4, 2x2, 1x1,
        'downsampling': ['c', 'sc', 'sc', 'sc'],
    },
    'small': {
        'img_size': 224,
        'patch_conv_type': '3conv',
        'patch_size': 4,
        'embed_dim': [96, 96, 192, 384, 768],
        'num_heads': [3, 6, 12, 24],
        'mlp_ratio': 4.,
        'depth': [2, 2, 8, 2],
        'kernel_sizes': [7, 7, 7, 7],  # 8x8, 4x4, 2x2, 1x1,
        'downsampling': ['c', 'sc', 'sc', 'sc'],
    },
    'medium': {
        'img_size': 224,
        'patch_conv_type': '1conv',
        'patch_size': 4,
        'embed_dim': [96] + [96 * (2 ** i) for i in range(4)],
        'num_heads': [3, 6, 12, 24],
        'mlp_ratio': 4.,
        'depth': [2, 2, 14, 2],
        'kernel_sizes': [7, 7, 7, 7],  # 8x8, 4x4, 2x2, 1x1,
        'downsampling': ['c', 'sc', 'sc', 'sc'],
    },
    'base': {
        'img_size': 224,
        'patch_conv_type': '1conv',
        'patch_size': 4,
        'embed_dim': [128, 128, 256, 512, 1024],
        'num_heads': [4, 8, 16, 32],
        'mlp_ratio': 4.,
        'depth': [2, 2, 14, 2],
        'kernel_sizes': [7, 7, 7, 7],  # 8x8, 4x4, 2x2, 1x1,
        'downsampling': ['c', 'sc', 'sc', 'sc'],
    },
    'small_w14': {
        'img_size': 224,
        'patch_conv_type': '3conv',
        'patch_size': 4,
        'embed_dim': [96, 96, 192, 384, 768],
        'num_heads': [3, 6, 12, 24],
        'mlp_ratio': 4.,
        'depth': [2, 2, 8, 2],
        'kernel_sizes': [14, 14, 14, 14],  # 8x8, 4x4, 2x2, 1x1,
        'downsampling': ['c', 'sc', 'sc', 'sc'],
    },
    'small_w14_peg': {
        'img_size': 224,
        'patch_conv_type': '3conv',
        'patch_size': 4,
        'embed_dim': [96, 96, 192, 384, 768],
        'num_heads': [3, 6, 12, 24],
        'mlp_ratio': 4.,
        'depth': [2, 2, 8, 2],
        'kernel_sizes': [14, 14, 14, 14],  # 8x8, 4x4, 2x2, 1x1,
        'downsampling': ['c', 'sc', 'sc', 'sc'],
        'peg': True
    },
    'base_w14': {
        'img_size': 224,
        'patch_conv_type': '1conv',
        'patch_size': 4,
        'embed_dim': [128, 128, 256, 512, 1024],
        'num_heads': [4, 8, 16, 32],
        'mlp_ratio': 4.,
        'depth': [2, 2, 14, 2],
        'kernel_sizes': [14, 14, 14, 14],  # 8x8, 4x4, 2x2, 1x1,
        'downsampling': ['c', 'sc', 'sc', 'sc'],
    },
    'base_w14_peg': {
        'img_size': 224,
        'patch_conv_type': '1conv',
        'patch_size': 4,
        'embed_dim': [128, 128, 256, 512, 1024],
        'num_heads': [4, 8, 16, 32],
        'mlp_ratio': 4.,
        'depth': [2, 2, 14, 2],
        'kernel_sizes': [14, 14, 14, 14],  # 8x8, 4x4, 2x2, 1x1,
        'downsampling': ['c', 'sc', 'sc', 'sc'],
        'peg': True
    },

}

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

def _regionvit(arch:str, pretrained:bool, progress:bool, **kwargs)->RegionViT:
    model_cfg = _model_cfg[arch]
    model = RegionViT(**model_cfg, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def regionvit_tiny_w14_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Regionvit-tiny-w14-224 model.

    .. note::
        Regionvit-tiny-w14-224 model from `"RegionViT: Regional-to-Local Attention for Vision Transformers" <https://arxiv.org/pdf/2106.02689.pdf>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> regionvit_tiny_w14_224 = flowvision.models.regionvit_tiny_w14_224(pretrained=False, progress=True)

    """
    return _regionvit('tiny_w14', pretrained, progress, **kwargs)


@ModelCreator.register_model
def regionvit_tiny_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Regionvit-tiny-224 model.

    .. note::
        Regionvit-tiny-224 model from `"RegionViT: Regional-to-Local Attention for Vision Transformers" <https://arxiv.org/pdf/2106.02689.pdf>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> regionvit_tiny_224 = flowvision.models.regionvit_tiny_224(pretrained=False, progress=True)

    """
    return _regionvit('tiny', pretrained, progress, **kwargs)


@ModelCreator.register_model
def regionvit_small_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Regionvit-small-224 model.

    .. note::
        Regionvit-small-224 model from `"RegionViT: Regional-to-Local Attention for Vision Transformers" <https://arxiv.org/pdf/2106.02689.pdf>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> regionvit_small_224 = flowvision.models.regionvit_small_224(pretrained=False, progress=True)

    """
    return _regionvit('small', pretrained, progress, **kwargs)


@ModelCreator.register_model
def regionvit_small_w14_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Regionvit-small-w14-224 model.

    .. note::
        Regionvit-small-w14-224 model from `"RegionViT: Regional-to-Local Attention for Vision Transformers" <https://arxiv.org/pdf/2106.02689.pdf>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> regionvit_small_w14_224 = flowvision.models.regionvit_small_w14_224(pretrained=False, progress=True)

    """
    return _regionvit('small_w14', pretrained, progress, **kwargs)


@ModelCreator.register_model
def regionvit_small_w14_peg_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Regionvit-small-w14-peg-224 model.

    .. note::
        Regionvit-small-w14-peg-224 model from `"RegionViT: Regional-to-Local Attention for Vision Transformers" <https://arxiv.org/pdf/2106.02689.pdf>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> regionvit_small_w14_peg_224 = flowvision.models.regionvit_small_w14_peg_224(pretrained=False, progress=True)

    """
    return _regionvit('small_w14_peg', pretrained, progress, **kwargs)


@ModelCreator.register_model
def regionvit_medium_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Regionvit-medium-224 model.

    .. note::
        Regionvit-medium-224 model from `"RegionViT: Regional-to-Local Attention for Vision Transformers" <https://arxiv.org/pdf/2106.02689.pdf>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> regionvit_medium_224 = flowvision.models.regionvit_medium_224(pretrained=False, progress=True)

    """
    return _regionvit('medium', pretrained, progress, **kwargs)


@ModelCreator.register_model
def regionvit_base_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Regionvit-base-224 model.

    .. note::
        Regionvit-base-224 model from `"RegionViT: Regional-to-Local Attention for Vision Transformers" <https://arxiv.org/pdf/2106.02689.pdf>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> regionvit_base_224 = flowvision.models.regionvit_base_224(pretrained=False, progress=True)

    """
    return _regionvit('base', pretrained, progress, **kwargs)


@ModelCreator.register_model
def regionvit_base_w14_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Regionvit-base-w14-224 model.

    .. note::
        Regionvit-base-w14-224 model from `"RegionViT: Regional-to-Local Attention for Vision Transformers" <https://arxiv.org/pdf/2106.02689.pdf>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> regionvit_base_w14_224 = flowvision.models.regionvit_base_w14_224(pretrained=False, progress=True)

    """
    return _regionvit('base_w14', pretrained, progress, **kwargs)


@ModelCreator.register_model
def regionvit_base_w14_peg_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Regionvit-base-w14-peg-224 model.

    .. note::
        Regionvit-base-w14-peg-224 model from `"RegionViT: Regional-to-Local Attention for Vision Transformers" <https://arxiv.org/pdf/2106.02689.pdf>`_.
        The required input size of the model is 224x224.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> regionvit_base_w14_peg_224 = flowvision.models.regionvit_base_w14_peg_224(pretrained=False, progress=True)

    """
    return _regionvit('base_w14_peg', pretrained, progress, **kwargs)