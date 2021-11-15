import oneflow.nn as nn
from .. import resnet
from .. import mobilenet_v3

from .seg_utils import _SimpleSegmentationModel, IntermediateLayerGetter
from ..utils import load_state_dict_from_url
from flowvision.models.registry import ModelCreator


model_urls = {
    "fcn_resnet50_coco": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/segmentation/FCN/fcn_resnet50_coco.zip",
    "fcn_resnet101_coco": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/segmentation/FCN/fcn_resnet101_coco.zip",
}


class FCN(_SimpleSegmentationModel):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.
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


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super(FCNHead, self).__init__(*layers)


def _fcn_segm_model(name, backbone_name, num_classes, aux, pretrained_backbone=True):
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

    classifier = FCNHead(out_inplanes, num_classes)
    base_model = FCN

    fcn_model = base_model(backbone, classifier, aux_classifier)
    return fcn_model


def _load_model(
    arch_type, backbone, pretrained, progress, num_classes, aux_loss, **kwargs
):
    if pretrained:
        aux_loss = True
        kwargs["pretrained_backbone"] = False
    model = _fcn_segm_model(arch_type, backbone, num_classes, aux_loss, **kwargs)
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
def fcn_resnet50_coco(
    pretrained=False, progress=True, num_classes=21, aux_loss=None, **kwargs
):
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model(
        "fcn", "resnet50", pretrained, progress, num_classes, aux_loss, **kwargs
    )


@ModelCreator.register_model
def fcn_resnet101_coco(
    pretrained=False, progress=True, num_classes=21, aux_loss=None, **kwargs
):
    """Constructs a Fully-Convolutional Network model with a ResNet-101 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model(
        "fcn", "resnet101", pretrained, progress, num_classes, aux_loss, **kwargs
    )
