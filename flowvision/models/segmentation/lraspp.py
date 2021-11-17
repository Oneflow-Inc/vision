from collections import OrderedDict

from oneflow import nn, Tensor
from oneflow.nn import functional as F
from typing import Dict

from .. import mobilenet_v3
from .seg_utils import _SimpleSegmentationModel, IntermediateLayerGetter
from flowvision.models.utils import load_state_dict_from_url
from flowvision.models.registry import ModelCreator


model_urls = {
    "lraspp_mobilenet_v3_large_coco": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/segmentation/lraspp/lraspp_mobilenet_v3_large_coco.zip",
}


class LRASPP(nn.Module):
    """
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int): number of output classes of the model (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.
    """

    def __init__(
        self, backbone, low_channels, high_channels, num_classes, inter_channels=128
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier = LRASPPHead(
            low_channels, high_channels, num_classes, inter_channels
        )

    def forward(self, input):
        features = self.backbone(input)
        out = self.classifier(features)
        out = F.interpolate(
            out, size=input.shape[-2:], mode="bilinear", align_corners=False
        )

        result = OrderedDict()
        result["out"] = out

        return result


class LRASPPHead(nn.Module):
    def __init__(self, low_channels, high_channels, num_classes, inter_channels):
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)


def _load_weights(model, arch_type, backbone, progress):
    arch = arch_type + "_" + backbone + "_coco"
    model_url = model_urls.get(arch, None)
    if model_url is None:
        raise NotImplementedError(
            "pretrained {} is not supported as of now".format(arch)
        )
    else:
        state_dict = load_state_dict_from_url(model_url, progress=progress)
        model.load_state_dict(state_dict)


def _segm_lraspp_mobilenetv3(backbone_name, num_classes, pretrained_backbone=True):
    backbone = mobilenet_v3.__dict__[backbone_name](
        pretrained=pretrained_backbone, dilated=True
    ).features

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = (
        [0]
        + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)]
        + [len(backbone) - 1]
    )
    low_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    high_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels

    backbone = IntermediateLayerGetter(
        backbone, return_layers={str(low_pos): "low", str(high_pos): "high"}
    )

    model = LRASPP(backbone, low_channels, high_channels, num_classes)
    return model


@ModelCreator.register_model
def lraspp_mobilenet_v3_large_coco(
    pretrained=False, progress=True, num_classes=21, **kwargs
):
    """Constructs a Lite R-ASPP Network model with a MobileNetV3-Large backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
    """
    if kwargs.pop("aux_loss", False):
        raise NotImplementedError("This model does not use auxiliary loss")

    backbone_name = "mobilenet_v3_large"
    model = _segm_lraspp_mobilenetv3(backbone_name, num_classes, **kwargs)

    if pretrained:
        _load_weights(model, "lraspp", backbone_name, progress)

    return model
