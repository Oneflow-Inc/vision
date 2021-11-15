import oneflow as flow
from oneflow import nn
from oneflow.nn import functional as F
from .fcn import FCNHead
from .. import resnet
from .. import mobilenet_v3

from .seg_utils import _SimpleSegmentationModel, IntermediateLayerGetter
from flowvision.models.utils import load_state_dict_from_url
from flowvision.models.registry import ModelCreator


model_urls = {
    "deeplabv3_resnet50_coco": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/segmentation/DeepLabV3/deeplabv3_resnet50_coco.zip",
    "deeplabv3_resnet101_coco": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/segmentation/DeepLabV3/deeplabv3_resnet101_coco.zip",
    "deeplabv3_mobilenet_v3_large_coco": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/segmentation/DeepLabV3/deeplabv3_mobilenet_v3_large_coco.zip",
}


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = flow.cat(res, dim=1)
        return self.project(res)


def _deeplab_segm_model(
    name, backbone_name, num_classes, aux, pretrained_backbone=True
):
    if "resnet" in backbone_name:
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=[False, True, True],
        )
        out_layer = "layer4"
        out_inplanes = 2048
        aux_layer = "layer3"
        aux_inplanes = 1024
    elif "mobilenet_v3" in backbone_name:
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
        out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
        out_layer = str(out_pos)
        out_inplanes = backbone[out_pos].out_channels
        aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
        aux_layer = str(aux_pos)
        aux_inplanes = backbone[aux_pos].out_channels
    else:
        raise NotImplementedError(
            "backbone {} is not supported as of now".format(backbone_name)
        )

    return_layers = {out_layer: "out"}
    if aux:
        return_layers[aux_layer] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = DeepLabHead(out_inplanes, num_classes)
    base_model = DeepLabV3

    deeplab_model = base_model(backbone, classifier, aux_classifier)
    return deeplab_model


def _load_model(
    arch_type, backbone, pretrained, progress, num_classes, aux_loss, **kwargs
):
    if pretrained:
        aux_loss = True
        kwargs["pretrained_backbone"] = False
    model = _deeplab_segm_model(arch_type, backbone, num_classes, aux_loss, **kwargs)
    if pretrained:
        _load_weights(model, arch_type, backbone, progress)
    return model


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


@ModelCreator.register_model
def deeplabv3_resnet50_coco(
    pretrained=False, progress=True, num_classes=21, aux_loss=None, **kwargs
):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model(
        "deeplabv3", "resnet50", pretrained, progress, num_classes, aux_loss, **kwargs
    )


@ModelCreator.register_model
def deeplabv3_resnet101_coco(
    pretrained=False, progress=True, num_classes=21, aux_loss=None, **kwargs
):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): The number of classes
        aux_loss (bool): If True, include an auxiliary classifier
    """
    return _load_model(
        "deeplabv3", "resnet101", pretrained, progress, num_classes, aux_loss, **kwargs
    )


@ModelCreator.register_model
def deeplabv3_mobilenet_v3_large_coco(
    pretrained=False, progress=True, num_classes=21, aux_loss=None, **kwargs
):
    """Constructs a DeepLabV3 model with a MobileNetV3-Large backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model(
        "deeplabv3",
        "mobilenet_v3_large",
        pretrained,
        progress,
        num_classes,
        aux_loss,
        **kwargs
    )
