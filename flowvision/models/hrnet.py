"""
Modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/hrnet.py
"""
from typing import Callable, List, Optional

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

from .registry import ModelCreator
from .utils import load_state_dict_from_url

_BN_MOMENTUM = 0.1

model_urls = {
    "hrnet_w18_small": "/dataset/ldl_home/Model/pretrained/hrnet_w18_small_v1-f460c6bc",
    "hrnet_w18_small_v2": "/dataset/ldl_home/Model/pretrained/hrnet_w18_small_v2-4c50a8cb",
    "hrnet_w18": "/dataset/ldl_home/Model/pretrained/hrnetv2_w18-8cb57bb9",
    "hrnet_w30": "/dataset/ldl_home/Model/pretrained/hrnetv2_w30-8d7f8dab",
    "hrnet_w32": "/dataset/ldl_home/Model/pretrained/hrnetv2_w32-90d8c5fb",
    "hrnet_w40": "/dataset/ldl_home/Model/pretrained/hrnetv2_w40-7cd397a4",
    "hrnet_w44": "/dataset/ldl_home/Model/pretrained/hrnetv2_w44-c9ac8c18",
    "hrnet_w48": "/dataset/ldl_home/Model/pretrained/hrnetv2_w48-abd2e6ab",
    "hrnet_w64": "/dataset/ldl_home/Model/pretrained/hrnetv2_w64-b47cc881"
}


cfg_cls = dict(
    hrnet_w18_small=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(1,),
            NUM_CHANNELS=(32,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2),
            NUM_CHANNELS=(16, 32),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2, 2),
            NUM_CHANNELS=(16, 32, 64),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2, 2, 2),
            NUM_CHANNELS=(16, 32, 64, 128),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w18_small_v2=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(2,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2),
            NUM_CHANNELS=(18, 36),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2, 2),
            NUM_CHANNELS=(18, 36, 72),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=2,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2, 2, 2),
            NUM_CHANNELS=(18, 36, 72, 144),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w18=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(18, 36),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(18, 36, 72),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(18, 36, 72, 144),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w30=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(30, 60),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(30, 60, 120),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(30, 60, 120, 240),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w32=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(32, 64),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(32, 64, 128),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(32, 64, 128, 256),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w40=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(40, 80),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(40, 80, 160),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(40, 80, 160, 320),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w44=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(44, 88),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(44, 88, 176),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(44, 88, 176, 352),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w48=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(48, 96),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(48, 96, 192),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(48, 96, 192, 384),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w64=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(64, 128),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(64, 128, 256),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(64, 128, 256, 512),
            FUSE_METHOD='SUM',
        ),
    )
)


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

    def forward(self, x):
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
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_in_chs,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_in_chs, num_channels)

        self.num_in_chs = num_in_chs
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.fuse_act = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks, num_in_chs, num_channels):
        error_msg = ''
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
        elif num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
        elif num_branches != len(num_in_chs):
            error_msg = 'NUM_BRANCHES({}) <> num_in_chs({})'.format(num_branches, len(num_in_chs))
        if error_msg:
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_in_chs[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_in_chs[branch_index], num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=_BN_MOMENTUM),
            )

        layers = [block(self.num_in_chs[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_in_chs[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_in_chs[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return nn.Identity()

        num_branches = self.num_branches
        num_in_chs = self.num_in_chs
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_in_chs[j], num_in_chs[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_in_chs[i], momentum=_BN_MOMENTUM),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_in_chs[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_in_chs[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=_BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_in_chs[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_in_chs[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=_BN_MOMENTUM),
                                nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_in_chs(self):
        return self.num_in_chs

    def forward(self, x: List[flow.Tensor]):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i, branch in enumerate(self.branches):
            x[i] = branch(x[i])

        x_fuse = []
        for i, fuse_outer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_outer[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + fuse_outer[j](x[j])
            x_fuse.append(self.fuse_act(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, in_chans=3, num_classes=1000, global_pool='avg', drop_rate=0.0, head='classification'):
        super(HighResolutionNet, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        stem_width = cfg['STEM_WIDTH']
        self.conv1 = nn.Conv2d(in_chans, stem_width, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_width, momentum=_BN_MOMENTUM)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(stem_width, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.act2 = nn.ReLU(inplace=True)

        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        self.head = head
        self.head_channels = None  # set if _make_head called
        if head == 'classification':
            # Classification Head
            self.num_features = 2048
            self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(pre_stage_channels)
            self.global_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1)
            )
            self.classifier = nn.Linear(self.num_features, self.num_classes, bias=True)
            # self.global_pool, self.classifier = create_classifier(
            #     self.num_features, self.num_classes, pool_type=global_pool)
        elif head == 'incre':
            self.num_features = 2048
            self.incre_modules, _, _ = self._make_head(pre_stage_channels, True)
        else:
            self.incre_modules = None
            self.num_features = 256

        curr_stride = 2
        # module names aren't actually valid here, hook or FeatureNet based extraction would not work
        self.feature_info = [dict(num_chs=64, reduction=curr_stride, module='stem')]
        for i, c in enumerate(self.head_channels if self.head_channels else num_channels):
            curr_stride *= 2
            c = c * 4 if self.head_channels else c  # head block expansion factor of 4
            self.feature_info += [dict(num_chs=c, reduction=curr_stride, module=f'stage{i + 1}')]

        self.init_weights()

    def _make_head(self, pre_stage_channels, incre_only=False):
        head_block = Bottleneck
        self.head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_modules.append(self._make_layer(head_block, channels, self.head_channels[i], 1, stride=1))
        incre_modules = nn.ModuleList(incre_modules)
        if incre_only:
            return incre_modules, None, None

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = self.head_channels[i] * head_block.expansion
            out_channels = self.head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.head_channels[3] * head_block.expansion,
                out_channels=self.num_features, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(self.num_features, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=_BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=_BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=_BN_MOMENTUM),
            )

        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_in_chs, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            reset_multi_scale_output = multi_scale_output or i < num_modules - 1
            modules.append(HighResolutionModule(
                num_branches, block, num_blocks, num_in_chs, num_channels, fuse_method, reset_multi_scale_output)
            )
            num_in_chs = modules[-1].get_num_in_chs()

        return nn.Sequential(*modules), num_in_chs

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^conv[12]|bn[12]',
            blocks=r'^(?:layer|stage|transition)(\d+)' if coarse else [
                (r'^layer(\d+)\.(\d+)', None),
                (r'^stage(\d+)\.(\d+)', None),
                (r'^transition(\d+)', (99999,)),
            ],
        )
        return matcher

    def set_grad_checkpointing(self, enable=True):
        assert not enable, "gradient checkpointing not supported"

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        self.classifier = nn.Linear(self.num_features, self.num_classes, bias=True)
        # self.global_pool, self.classifier = create_classifier(
        #     self.num_features, self.num_classes, pool_type=global_pool)

    def stages(self, x) -> List[flow.Tensor]:
        x = self.layer1(x)

        xl = [t(x) for i, t in enumerate(self.transition1)]
        yl = self.stage2(xl)

        xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(self.transition2)]
        yl = self.stage3(xl)

        xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(self.transition3)]
        yl = self.stage4(xl)
        return yl

    def forward_features(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Stages
        yl = self.stages(x)
        if self.incre_modules is None or self.downsamp_modules is None:
            return yl
        y = self.incre_modules[0](yl[0])
        for i, down in enumerate(self.downsamp_modules):
            y = self.incre_modules[i + 1](yl[i + 1]) + down(y)
        y = self.final_layer(y)
        return y

    def forward_head(self, x, pre_logits: bool = False):
        # Classification Head
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        y = self.forward_features(x)
        x = self.forward_head(y)
        return x


# 13.187464M
# Acc@1: 72.078, Acc@1-Error: 27.922, Acc@5: 90.470, Acc@5-Error: 9.530
@ModelCreator.register_model
def hrnet_w18_small(pretrained: bool = False, progress: bool = True, **kwargs):
    arch = 'hrnet_w18_small'
    model = HighResolutionNet(cfg=cfg_cls[arch], **kwargs)
    if pretrained:
        state_dict = flow.load(model_urls[arch])
        # state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

# 15.597464M
# Acc@1: 74.816, Acc@1-Error: 25.184, Acc@5: 92.264, Acc@5-Error: 7.736
@ModelCreator.register_model
def hrnet_w18_small_v2(pretrained: bool = False, progress: bool = True, **kwargs):
    arch = 'hrnet_w18_small_v2'
    model = HighResolutionNet(cfg=cfg_cls[arch], **kwargs)
    if pretrained:
        state_dict = flow.load(model_urls[arch])
        # state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

# 21.299004M
# Acc@1: 76.484, Acc@1-Error: 23.516, Acc@5: 93.322, Acc@5-Error: 6.678
@ModelCreator.register_model
def hrnet_w18(pretrained: bool = False, progress: bool = True, **kwargs):
    arch = 'hrnet_w18'
    model = HighResolutionNet(cfg=cfg_cls[arch], **kwargs)
    if pretrained:
        state_dict = flow.load(model_urls[arch])
        # state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

# 37.71222M
# Acc@1: 78.074, Acc@1-Error: 21.926, Acc@5: 94.136, Acc@5-Error: 5.864
@ModelCreator.register_model
def hrnet_w30(pretrained: bool = False, progress: bool = True, **kwargs):
    arch = 'hrnet_w30'
    model = HighResolutionNet(cfg=cfg_cls[arch], **kwargs)
    if pretrained:
        state_dict = flow.load(model_urls[arch])
        # state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

# 41.23268M
# Acc@1: 78.188, Acc@1-Error: 21.812, Acc@5: 94.034, Acc@5-Error: 5.966
@ModelCreator.register_model
def hrnet_w32(pretrained: bool = False, progress: bool = True, **kwargs):
    arch = 'hrnet_w32'
    model = HighResolutionNet(cfg=cfg_cls[arch], **kwargs)
    if pretrained:
        state_dict = flow.load(model_urls[arch])
        # state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

# 57.55716M
# Acc@1: 78.652, Acc@1-Error: 21.348, Acc@5: 94.378, Acc@5-Error: 5.622
@ModelCreator.register_model
def hrnet_w40(pretrained: bool = False, progress: bool = True, **kwargs):
    arch = 'hrnet_w40'
    model = HighResolutionNet(cfg=cfg_cls[arch], **kwargs)
    if pretrained:
        state_dict = flow.load(model_urls[arch])
        # state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

# 67.064984M
# Acc@1: 78.646, Acc@1-Error: 21.354, Acc@5: 94.250, Acc@5-Error: 5.750
@ModelCreator.register_model
def hrnet_w44(pretrained: bool = False, progress: bool = True, **kwargs):
    arch = 'hrnet_w44'
    model = HighResolutionNet(cfg=cfg_cls[arch], **kwargs)
    if pretrained:
        state_dict = flow.load(model_urls[arch])
        # state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

# 77.469864M
# Acc@1: 78.972, Acc@1-Error: 21.028, Acc@5: 94.344, Acc@5-Error: 5.656
@ModelCreator.register_model
def hrnet_w48(pretrained: bool = False, progress: bool = True, **kwargs):
    arch = 'hrnet_w48'
    model = HighResolutionNet(cfg=cfg_cls[arch], **kwargs)
    if pretrained:
        state_dict = flow.load(model_urls[arch])
        # state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

# 128.059944M
# Acc@1: 79.210, Acc@1-Error: 20.790, Acc@5: 94.558, Acc@5-Error: 5.442
@ModelCreator.register_model
def hrnet_w64(pretrained: bool = False, progress: bool = True, **kwargs):
    arch = 'hrnet_w64'
    model = HighResolutionNet(cfg=cfg_cls[arch], **kwargs)
    if pretrained:
        state_dict = flow.load(model_urls[arch])
        # state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model
