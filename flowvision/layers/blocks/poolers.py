from typing import Optional, List, Dict, Tuple, Union

import oneflow as flow
from oneflow import nn, Tensor
from flowvision.layers.blocks.boxes import box_area

from .roi_align import roi_align


def initLevelMapper(
    k_min: int,
    k_max: int,
    canonical_scale: int = 224,
    canonical_level: int = 4,
    eps: float = 1e-6,
):
    return LevelMapper(k_min, k_max, canonical_scale, canonical_level, eps)


class LevelMapper:
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.

    Args:
        k_min (int)
        k_max (int)
        canonical_scale (int)
        canonical_level (int)
        eps (float)
    """

    def __init__(
        self,
        k_min: int,
        k_max: int,
        canonical_scale: int = 224,
        canonical_level: int = 4,
        eps: float = 1e-6,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists: List[Tensor]) -> Tensor:
        """
        Args:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = flow.sqrt(flow.cat([box_area(boxlist) for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = flow.floor(
            self.lvl0
            + flow.log2(s / self.s0)
            + flow.tensor(self.eps, dtype=s.dtype, device=s.device)
        )
        target_lvls = flow.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return (target_lvls.to(flow.int64) - self.k_min).to(flow.int64)


def _convert_to_roi_format(boxes: List[Tensor]) -> Tensor:
    concat_boxes = flow.cat(boxes, dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = flow.cat(
        [
            flow.full(b[:, :1].shape, i, dtype=dtype, device=device)
            for i, b in enumerate(boxes)
        ],
        dim=0,
    )
    rois = flow.cat([ids, concat_boxes], dim=1)
    return rois


def _infer_scale(feature: Tensor, original_size: List[int]) -> float:
    # assumption: the scale is of the form 2 ** (-k), with k integer
    size = feature.shape[-2:]
    possible_scales: List[float] = []
    for s1, s2 in zip(size, original_size):
        approx_scale = float(s1) / float(s2)
        scale = 2 ** float(flow.tensor(approx_scale).log2().round().item())
        possible_scales.append(scale)
    assert possible_scales[0] == possible_scales[1]
    return possible_scales[0]


def _setup_scales(
    features: List[Tensor],
    image_shapes: List[Tuple[int, int]],
    canonical_scale: int,
    canonical_level: int,
) -> Tuple[List[float], LevelMapper]:
    assert len(image_shapes) != 0
    max_x = 0
    max_y = 0
    for shape in image_shapes:
        max_x = max(shape[0], max_x)
        max_y = max(shape[1], max_y)
    original_input_shape = (max_x, max_y)

    scales = [_infer_scale(feat, original_input_shape) for feat in features]
    # get the levels in the feature map by leveraging the fact that the network always
    # downsamples by a factor of 2 at each level.
    lvl_min = -flow.log2(flow.tensor(scales[0], dtype=flow.float32)).item()
    lvl_max = -flow.log2(flow.tensor(scales[-1], dtype=flow.float32)).item()

    map_levels = initLevelMapper(
        int(lvl_min),
        int(lvl_max),
        canonical_scale=canonical_scale,
        canonical_level=canonical_level,
    )
    return scales, map_levels


def _filter_input(x: Dict[str, Tensor], featmap_names: List[str]) -> List[Tensor]:
    x_filtered = []
    for k, v in x.items():
        if k in featmap_names:
            x_filtered.append(v)
    return x_filtered


def _multiscale_roi_align(
    x_filtered: List[Tensor],
    boxes: List[Tensor],
    output_size: List[int],
    sampling_ratio: int,
    scales: Optional[List[float]],
    mapper: Optional[LevelMapper],
) -> Tensor:
    """
    Args:
        x_filtered (List[Tensor]): List of input tensors.
        boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
            (x1, y1, x2, y2) format and in the image reference size, not the feature map
            reference. The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        output_size (Union[List[Tuple[int, int]], List[int]]): size of the output
        sampling_ratio (int): sampling ratio for ROIAlign
        scales (Optional[List[float]]): If None, scales will be automatically infered. Default value is None.
        mapper (Optional[LevelMapper]): If None, mapper will be automatically infered. Default value is None.
    Returns:
        result (Tensor)
    """
    assert scales is not None
    assert mapper is not None

    num_levels = len(x_filtered)
    rois = _convert_to_roi_format(boxes)

    if num_levels == 1:
        return roi_align(
            x_filtered[0],
            rois,
            output_size=output_size,
            spatial_scale=scales[0],
            sampling_ratio=sampling_ratio,
        )

    levels = mapper(boxes)

    num_rois = len(rois)
    num_channels = x_filtered[0].shape[1]

    dtype, device = x_filtered[0].dtype, x_filtered[0].device
    result = flow.zeros(
        (num_rois, num_channels,) + output_size, dtype=dtype, device=device,
    )

    for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
        idx_in_level = flow.where(levels == level)[0]
        rois_per_level = rois[idx_in_level]

        result_idx_in_level = roi_align(
            per_level_feature,
            rois_per_level,
            output_size=output_size,
            spatial_scale=scale,
            sampling_ratio=sampling_ratio,
        )
        # TODO (shijie wang): Use cmobined indexing.
        # result[idx_in_level] = result_idx_in_level.to(result.dtype)
        result = flow.tensor_scatter_nd_update(
            result, idx_in_level[:, None], result_idx_in_level
        )

    return result


class MultiScaleRoIAlign(nn.Module):
    """
    Multi-scale RoIAlign pooling, which is useful for detection with or without FPN.

    It infers the scale of the pooling via the heuristics specified in eq. 1
    of the `Feature Pyramid Network paper <https://arxiv.org/abs/1612.03144>`_.
    They keyword-only parameters ``canonical_scale`` and ``canonical_level``
    correspond respectively to ``224`` and ``k0=4`` in eq. 1, and
    have the following meaning: ``canonical_level`` is the target level of the pyramid from
    which to pool a region of interest with ``w x h = canonical_scale x canonical_scale``.

    Args:
        featmap_names (List[str]): the names of the feature maps that will be used
            for the pooling.
        output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
        sampling_ratio (int): sampling ratio for ROIAlign
        canonical_scale (int, optional): canonical_scale for LevelMapper
        canonical_level (int, optional): canonical_level for LevelMapper
    """

    __annotations__ = {
        "scales": Optional[List[float]],
        "map_levels": Optional[LevelMapper],
    }

    def __init__(
        self,
        featmap_names: List[str],
        output_size: Union[int, Tuple[int], List[int]],
        sampling_ratio: int,
        *,
        canonical_scale: int = 224,
        canonical_level: int = 4,
    ):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = featmap_names
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None
        self.map_levels = None
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level

    def forward(
        self,
        x: Dict[str, Tensor],
        boxes: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tensor:
        """
        Args:
            x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
                (x1, y1, x2, y2) format and in the image reference size, not the feature map
                reference. The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            image_shapes (List[Tuple[height, width]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """
        x_filtered = _filter_input(x, self.featmap_names)
        if self.scales is None or self.map_levels is None:
            self.scales, self.map_levels = _setup_scales(
                x_filtered, image_shapes, self.canonical_scale, self.canonical_level
            )

        return _multiscale_roi_align(
            x_filtered,
            boxes,
            self.output_size,
            self.sampling_ratio,
            self.scales,
            self.map_levels,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(featmap_names={self.featmap_names}, "
            f"output_size={self.output_size}, sampling_ratio={self.sampling_ratio})"
        )
