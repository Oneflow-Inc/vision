import oneflow as flow
from oneflow import nn
import oneflow.nn.functional as F

from flowvision.layers.blocks import MultiScaleRoIAlign
from flowvision.layers.blocks import misc as misc_nn_ops

from .det_utils import overwrite_eps
from ..utils import load_state_dict_from_url
from ..registry import ModelCreator

from ..resnet import resnet50
from ..mobilenet_v3 import mobilenet_v3_large
from .anchor_utils import AnchorGenerator
from .generalized_rcnn import GeneralizedRCNN
from .rpn import RPNHead, RegionProposalNetwork
from .roi_heads import RoIHeads
from .transform import GeneralizedRCNNTransform
from .backbone_utils import (
    _resnet_fpn_extractor,
    _validate_trainable_layers,
    _mobilenet_extractor,
)


__all__ = ["FasterRCNN", "fasterrcnn_resnet50_fpn"]


model_urls = {
    "fasterrcnn_resnet50_fpn_coco": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/detection/faster_rcnn/fasterrcnn_resnet50_fpn_coco.tar.gz",
    "fasterrcnn_mobilenet_v3_large_320_fpn_coco": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/detection/faster_rcnn/fasterrcnn_mobilenet_v3_large_320_fpn.tar.gz",
    "fasterrcnn_mobilenet_v3_large_fpn_coco": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/detection/faster_rcnn/fasterrcnn_mobilenet_v3_large_fpn.tar.gz",
}


class FasterRCNN(GeneralizedRCNN):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone.
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone.
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained on.
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on.
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN.
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training.
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing.
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training.
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing.
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss.
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN.
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh.
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes.
        box_head (nn.Module): module that takes the cropped feature maps as input.
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh.
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference.
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head.
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head.
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head.
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head.
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes.
    """

    def __init__(
        self,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
    ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError(
                    "num_classes should be None when box_predictor is specified"
                )
        else:
            if box_predictor is None:
                raise ValueError(
                    "num_classes should not be None when box_predictor "
                    "is not specified"
                )

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test
        )
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test
        )

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
            )

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super().__init__(backbone, rpn, roi_heads, transform)


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


@ModelCreator.register_model
def fasterrcnn_resnet50_fpn(
    pretrained=False,
    progress=True,
    num_classes=91,
    pretrained_backbone=True,
    trainable_backbone_layers=None,
    **kwargs,
):
    """
    Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.

    Reference: `"Faster R-CNN: Towards Real-Time Object Detection with
    Region Proposal Networks" <https://arxiv.org/abs/1506.01497>`_.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    images, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the psot-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 3.
    """
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
    )

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet50(
        pretrained=pretrained_backbone,
        progress=progress,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    )
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["fasterrcnn_resnet50_fpn_coco"], progress=progress
        )
        model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)
    return model


def _fasterrcnn_mobilenet_v3_large_fpn(
    weights_name,
    pretrained=False,
    progress=True,
    num_classes=91,
    pretrained_backbone=True,
    trainable_backbone_layers=None,
    **kwargs,
):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 3
    )

    if pretrained:
        pretrained_backbone = False

    backbone = mobilenet_v3_large(
        pretrained=pretrained_backbone,
        progress=progress,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    )
    backbone = _mobilenet_extractor(backbone, True, trainable_backbone_layers)

    anchor_sizes = ((32, 64, 128, 256, 512,),) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    model = FasterRCNN(
        backbone,
        num_classes,
        rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
        **kwargs,
    )
    if pretrained:
        if model_urls.get(weights_name, None) is None:
            raise ValueError(f"No checkpoint is available for model {weights_name}")
        state_dict = load_state_dict_from_url(
            model_urls[weights_name], progress=progress
        )
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def fasterrcnn_mobilenet_v3_large_320_fpn(
    pretrained=False,
    progress=True,
    num_classes=91,
    pretrained_backbone=True,
    trainable_backbone_layers=None,
    **kwargs,
):
    """
    Constructs a low resolution Faster R-CNN model with a MobileNetV3-Large FPN backbone tunned for mobile use-cases.
    It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See
    :func:`~flowvision.models.detection.fasterrcnn_resnet50_fpn` for more
    details.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 6, with 6 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 3.
    """
    weights_name = "fasterrcnn_mobilenet_v3_large_320_fpn_coco"
    defaults = {
        "min_size": 320,
        "max_size": 640,
        "rpn_pre_nms_top_n_test": 150,
        "rpn_post_nms_top_n_test": 150,
        "rpn_score_thresh": 0.05,
    }

    kwargs = {**defaults, **kwargs}
    return _fasterrcnn_mobilenet_v3_large_fpn(
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs,
    )


@ModelCreator.register_model
def fasterrcnn_mobilenet_v3_large_fpn(
    pretrained=False,
    progress=True,
    num_classes=91,
    pretrained_backbone=True,
    trainable_backbone_layers=None,
    **kwargs,
):
    """
    Constructs a high resolution Faster R-CNN model with a MobileNetV3-Large FPN backbone.
    It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See
    :func:`~flowvision.models.detection.fasterrcnn_resnet50_fpn` for more
    details.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 6, with 6 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 3.
    """
    weights_name = "fasterrcnn_mobilenet_v3_large_fpn_coco"
    defaults = {
        "rpn_score_thresh": 0.05,
    }

    kwargs = {**defaults, **kwargs}
    return _fasterrcnn_mobilenet_v3_large_fpn(
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs,
    )
