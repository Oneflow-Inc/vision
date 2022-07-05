"""
Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/generalized_rcnn.py
"""
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union

import oneflow as flow
from oneflow import nn, Tensor


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): take the features + the proposals from the RPN and computes
            detections / makes from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(
        self,
        backbone: nn.Module,
        rpn: nn.Module,
        roi_heads: nn.Module,
        transform: nn.Module,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the images (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targes should be passed")

            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, flow.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}."
                        )
                else:
                    raise ValueError(
                        f"Expected target boxes to be of type Tensor, got {type(boxes)}."
                    )

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            if len(val) != 2:
                raise ValueError(
                    f"Expecting the last two dimensions of the input tensor to be H and W, instead got {img.shape[-2:]}"
                )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = flow.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, flow.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections
