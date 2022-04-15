import oneflow as flow
from oneflow import Tensor


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.dtype not in (flow.float32, flow.float64):
        return t.float()
    return t


def generalized_box_iou_loss(
    boxes1: Tensor, boxes2: Tensor, reduction: str = "none", eps: float = 1e-7,
) -> Tensor:
    """
    Original implementation from
    https://github.com/facebookresearch/fvcore/blob/bfff2ef/fvcore/nn/giou_loss.py

    Gradient-friendly IoU loss with an additional penalty that is non-zero when the
    boxes do not overlap and scales with the size of their smallest enclosing box.
    This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``, and The two boxes should have the
    same dimensions.

    Args:
        boxes1 (Tensor[N, 4] or Tensor[4]): first set of boxes
        boxes2 (Tensor[N, 4] or Tensor[4]): second set of boxes
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: No reduction will be
            applied to the output. ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'none'``
        eps (float, optional): small number to prevent division by zero. Default: 1e-7

    Reference:
        Hamid Rezatofighi et. al: Generalized Intersection over Union:
        A Metric and A Loss for Bounding Box Regression:
        https://arxiv.org/abs/1902.09630
    """
    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # Intersection keypoints
    xkis1 = flow.max(x1, x1g)
    ykis1 = flow.max(y1, y1g)
    xkis2 = flow.min(x2, x2g)
    ykis2 = flow.min(y2, y2g)

    # TODO (shijie wang): fix the bug of tensor_scatter_nd_update_backward
    # intsctk = flow.zeros_like(x1)
    # mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    # intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    intsctk = (ykis2 - ykis1).clamp(min=0) * (xkis2 - xkis1).clamp(min=0)
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)

    # smallest enclosing box
    xc1 = flow.min(x1, x1g)
    yc1 = flow.min(y1, y1g)
    xc2 = flow.max(x2, x2g)
    yc2 = flow.max(y2, y2g)

    area_c = (xc2 - xc1) * (yc2 - yc1)
    miouk = iouk - ((area_c - unionk) / (area_c + eps))

    loss = 1 - miouk

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
