"""
"""
import warnings
from typing import Optional, Tuple, List

from oneflow.framework.tensor import Tensor
import oneflow as flow


def _is_tensor_a_flow_image(x: Tensor) -> bool:
    return x.ndim >= 2


def _assert_image_tensor(img):
    if not _is_tensor_a_flow_image(img):
        raise TypeError("Tensor is not a flow image.")


def _get_image_size(img: Tensor) -> List[int]:
    # Returns (w, h) of tensor image
    _assert_image_tensor(img)
    return [img.shape[-1], img.shape[-2]]


def _get_image_num_channels(img: Tensor) -> int:
    if img.ndim == 2:
        return 1
    elif img.ndim > 2:
        return img.shape[-3]

    raise TypeError("Input ndim should be 2 or more. Got {}".format(img.ndim))


def _max_value(dtype: flow.dtype) -> float:

    a = flow.tensor(2, dtype=dtype)
    # TODO:Tensor.is_signed()
    # signed = 1 if flow.tensor(0, dtype=dtype).is_signed() else 0
    signed = 1
    bits = 1
    max_value = flow.tensor(-signed, dtype=flow.long)
    while True:
        next_value = a.pow(bits - signed).sub(1)
        if next_value > max_value:
            max_value = next_value
            bits *= 2
        else:
            break
    return max_value.item()


def _assert_channels(img: Tensor, permitted: List[int]) -> None:
    c = _get_image_num_channels(img)
    if c not in permitted:
        raise TypeError(
            "Input image tensor permitted channel values are {}, but found {}".format(
                permitted, c
            )
        )


def _cast_squeeze_in(
    img: Tensor, req_dtypes: List[flow.dtype]
) -> Tuple[Tensor, bool, bool, flow.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = flow._C.cast(img, req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(
    img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: flow.dtype
):
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (flow.uint8, flow.int8, flow.int16, flow.int32, flow.int64):
            # it is better to round before cast
            img = flow.round(img)
        img = flow._C.cast(img, out_dtype)
    return img


def convert_image_dtype(
    image: flow.Tensor, dtype: flow.dtype = flow.float
) -> flow.Tensor:
    if image.dtype == dtype:
        return image

    if image.is_floating_point():
        # TODO:Tensor.is_floating_point()
        if flow.tensor(0, dtype=dtype).is_floating_point():
            return flow._C.cast(image, dtype)

        # float to int
        if (image.dtype == flow.float32 and dtype in (flow.int32, flow.int64)) or (
            image.dtype == flow.float64 and dtype == flow.int64
        ):
            msg = f"The cast from {image.dtype} to {dtype} cannot be performed safely."
            raise RuntimeError(msg)

        # https://github.com/pytorch/vision/pull/2078#issuecomment-612045321
        # For data in the range 0-1, (float * 255).to(uint) is only 255
        # when float is exactly 1.0.
        # `max + 1 - epsilon` provides more evenly distributed mapping of
        # ranges of floats to ints.
        eps = 1e-3
        max_val = _max_value(dtype)
        result = image.mul(max_val + 1.0 - eps)
        return flow._C.cast(result, dtype)
    else:
        input_max = _max_value(image.dtype)

        # int to float
        if flow.tensor(0, dtype=dtype).is_floating_point():
            image = flow._C.cast(image, dtype)
            return image / input_max

        output_max = _max_value(dtype)

        # int to int
        if input_max > output_max:
            factor = int((input_max + 1) // (output_max + 1))
            image = flow.div(image, factor, rounding_mode="floor")
            return flow._C.cast(image, dtype)
        else:
            factor = int((output_max + 1) // (input_max + 1))
            image = flow._C.cast(image, dtype)
            return image * factor


def vflip(img: Tensor) -> Tensor:
    _assert_image_tensor(img)

    return img.flip(-2)


def hflip(img: Tensor) -> Tensor:
    _assert_image_tensor(img)

    return img.flip(-1)


def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    _assert_image_tensor(img)

    w, h = _get_image_size(img)
    right = left + width
    bottom = top + height

    if left < 0 or top < 0 or right > w or bottom > h:
        padding_ltrb = [
            max(-left, 0),
            max(-top, 0),
            max(right - w, 0),
            max(bottom - h, 0),
        ]
        return pad(
            img[..., max(top, 0) : bottom, max(left, 0) : right], padding_ltrb, fill=0
        )
    return img[..., top:bottom, left:right]


def rgb_to_grayscale(img: Tensor, num_output_channels: int = 1) -> Tensor:
    if img.ndim < 3:
        raise TypeError(
            "Input image tensor should have at least 3 dimensions, but found {}".format(
                img.ndim
            )
        )
    _assert_channels(img, [3])

    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels should be either 1 or 3")

    r, g, b = img.unbind(dim=-3)
    # This implementation closely follows the TF one:
    # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img


def adjust_brightness(img: Tensor, brightness_facotr: float) -> Tensor:
    if brightness_facotr < 0:
        raise ValueError(
            "brightness_factor ({}) is not non-negative.".format(brightness_facotr)
        )

    _assert_image_tensor(img)

    _assert_channels(img, [1, 3])

    return _blend(img, flow.zeros_like(img), brightness_facotr)


def adjust_contrast(img: Tensor, contrast_factor: float) -> Tensor:
    if contrast_factor < 0:
        raise ValueError(
            "contrast_factor ({}) is not non-negative.".format(contrast_factor)
        )

    _assert_image_tensor(img)

    _assert_channels(img, [3])

    dtype = img.dtype if flow.is_floating_point(img) else flow.float32
    mean = flow.mean(rgb_to_grayscale(img).to(dtype), dim=(-3, -2, -1), keepdim=True)

    return _blend(img, mean, contrast_factor)


def adjust_hue(img: Tensor, hue_factor: float) -> Tensor:
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError("hue_factor ({}) is not in [-0.5, 0.5].".format(hue_factor))

    if not (isinstance(img, flow.Tensor)):
        raise TypeError("Input img should be Tensor image")

    _assert_image_tensor(img)

    _assert_channels(img, [1, 3])
    if _get_image_num_channels(img) == 1:  # Match PIL behaviour
        return img

    orig_dtype = img.dtype
    if img.dtype == flow.uint8:
        img = img.to(dtype=flow.float32) / 255.0

    img = _rgb2hsv(img)
    h, s, v = img.unbind(dim=-3)
    h = (h + hue_factor) % 1.0
    img = flow.stack((h, s, v), dim=-3)
    img_hue_adj = _hsv2rgb(img)

    if orig_dtype == flow.uint8:
        img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)

    return img_hue_adj


def adjust_saturation(img: Tensor, saturation_factor: float) -> Tensor:
    if saturation_factor < 0:
        raise ValueError(
            "saturation_factor ({}) is not non-negative.".format(saturation_factor)
        )

    _assert_image_tensor(img)

    _assert_channels(img, [3])

    return _blend(img, rgb_to_grayscale(img), saturation_factor)


def _blend(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    ratio = float(ratio)
    bound = 1.0 if img1.is_floating_point() else 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def _rgb2hsv(img: Tensor) -> Tensor:
    r, g, b = img.unbind(dim=-3)

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = flow.max(img, dim=-3).values
    minc = flow.min(img, dim=-3).values

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = flow.ones_like(maxc)
    s = cr / flow.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = flow.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = flow.fmod((h / 6.0 + 1.0), 1.0)
    return flow.stack((h, s, maxc), dim=-3)


def _hsv2rgb(img: Tensor) -> Tensor:
    h, s, v = img.unbind(dim=-3)
    i = flow.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=flow.int32)

    p = flow.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = flow.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = flow.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = i.unsqueeze(dim=-3) == flow.arange(6, device=i.device).view(-1, 1, 1)

    a1 = flow.stack((v, q, p, p, t, v), dim=-3)
    a2 = flow.stack((t, v, v, q, p, p), dim=-3)
    a3 = flow.stack((p, p, t, v, v, q), dim=-3)
    a4 = flow.stack((a1, a2, a3), dim=-4)

    return flow.sum(mask.to(dtype=img.dtype).unsqueeze(0) * a4, dim=1)


def _pad_symmetric(img: Tensor, padding: List[int]) -> Tensor:
    # padding is left, right, top, bottom

    # crop if needed
    if padding[0] < 0 or padding[1] < 0 or padding[2] < 0 or padding[3] < 0:
        crop_left, crop_right, crop_top, crop_bottom = [-min(x, 0) for x in padding]
        img = img[
            ...,
            crop_top : img.shape[-2] - crop_bottom,
            crop_left : img.shape[-1] - crop_right,
        ]
        padding = [max(x, 0) for x in padding]

    in_sizes = img.size()

    x_indices = [i for i in range(in_sizes[-1])]  # [0, 1, 2, 3, ...]
    left_indices = [i for i in range(padding[0] - 1, -1, -1)]  # e.g. [3, 2, 1, 0]
    right_indices = [-(i + 1) for i in range(padding[1])]  # e.g. [-1, -2, -3]
    x_indices = flow.tensor(left_indices + x_indices + right_indices, device=img.device)

    y_indices = [i for i in range(in_sizes[-2])]
    top_indices = [i for i in range(padding[2] - 1, -1, -1)]
    bottom_indices = [-(i + 1) for i in range(padding[3])]
    y_indices = flow.tensor(top_indices + y_indices + bottom_indices, device=img.device)

    ndim = img.ndim
    if ndim == 3:
        return img[:, y_indices[:, None], x_indices[None, :]]
    elif ndim == 4:
        return img[:, :, y_indices[:, None], x_indices[None, :]]
    else:
        raise RuntimeError("Symmetric padding of N-D tensors are not supported yet")


def pad(
    img: Tensor, padding: List[int], fill: int = 0, padding_mode: str = "constant"
) -> Tensor:
    _assert_image_tensor(img)

    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (int, float)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, tuple):
        padding = list(padding)

    if isinstance(padding, list) and len(padding) not in [1, 2, 4]:
        raise ValueError(
            "Padding must be an int or a 1, 2, or 4 element tuple, not a "
            + "{} element tuple".format(len(padding))
        )

    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError(
            "Padding mode should be either constant, edge, reflect or symmetric"
        )

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 1:
        pad_left = pad_right = pad_top = pad_bottom = padding[0]
    elif len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    else:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    p = [pad_left, pad_right, pad_top, pad_bottom]

    if padding_mode == "edge":
        # remap padding_mode str
        padding_mode = "replicate"
    elif padding_mode == "symmetric":
        # route to another implementation
        return _pad_symmetric(img, p)

    need_squeeze = False
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if (padding_mode != "constant") and img.dtype not in (flow.float32, flow.float64):
        # Here we temporary cast input tensor to float
        # until pytorch issue is resolved :
        # https://github.com/pytorch/pytorch/issues/40763
        need_cast = True
        img = flow._C.cast(img, flow.float32)
    img = flow._C.pad(img, pad=p, mode=padding_mode, value=float(fill))

    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        img = flow._C.cast(img, out_dtype)
    return img


def resize(img: Tensor, size: List[int], interpolation: str = "bilinear") -> Tensor:
    _assert_image_tensor(img)

    if not isinstance(size, (int, tuple, list)):
        raise TypeError("Got inappropriate size arg")
    if not isinstance(interpolation, str):
        raise TypeError("Got inappropriate interpolation arg")

    if interpolation not in ["nearest", "bilinear", "bicubic"]:
        raise ValueError("This interpolation mode is unsupported with Tensor input")

    if isinstance(size, tuple):
        size = list(size)

    if isinstance(size, list) and len(size) not in [1, 2]:
        raise ValueError(
            "Size must be an int or a 1 or 2 element tuple/list, not a "
            "{} element tuple/list".format(len(size))
        )

    w, h = _get_image_size(img)

    if isinstance(size, int):
        size_w, size_h = size, size
    elif len(size) < 2:
        size_w, size_h = size[0], size[0]
    else:
        size_w, size_h = size[1], size[0]  # Convention (h, w)

    if isinstance(size, int) or len(size) < 2:
        if w < h:
            size_h = int(size_w * h / w)
        else:
            size_w = int(size_h * w / h)

        if (w <= h and w == size_w) or (h <= w and h == size_h):
            return img

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(
        img, [flow.float32, flow.float64]
    )

    # Define align_corners to avoid warnings
    align_corners = False if interpolation in ["bilinear", "bicubic"] else None

    img = flow.nn.functional.interpolate(
        img, size=[size_h, size_w], mode=interpolation, align_corners=align_corners
    )

    if interpolation == "bicubic" and out_dtype == flow.uint8:
        img = img.clamp(min=0, max=255)

    img = _cast_squeeze_out(
        img, need_cast=need_cast, need_squeeze=need_squeeze, out_dtype=out_dtype
    )

    return img


def _assert_grid_transform_inputs(
    img: Tensor,
    matrix: Optional[List[float]],
    interpolation: str,
    fill: Optional[List[float]],
    supported_interpolation_modes: List[str],
    coeffs: Optional[List[float]] = None,
):

    if not (isinstance(img, flow.Tensor)):
        raise TypeError("Input img should be Tensor")

    _assert_image_tensor(img)

    if matrix is not None and not isinstance(matrix, list):
        raise TypeError("Argument matrix should be a list")

    if matrix is not None and len(matrix) != 6:
        raise ValueError("Argument matrix should have 6 float values")

    if coeffs is not None and len(coeffs) != 8:
        raise ValueError("Argument coeffs should have 8 float values")

    if fill is not None and not isinstance(fill, (int, float, tuple, list)):
        warnings.warn("Argument fill should be either int, float, tuple or list")

    # Check fill
    num_channels = _get_image_num_channels(img)
    if isinstance(fill, (tuple, list)) and (
        len(fill) > 1 and len(fill) != num_channels
    ):
        msg = (
            "The number of elements in 'fill' cannot broadcast to match the number of "
            "channels of the image ({} != {})"
        )
        raise ValueError(msg.format(len(fill), num_channels))

    if interpolation not in supported_interpolation_modes:
        raise ValueError(
            "Interpolation mode '{}' is unsupported with Tensor input".format(
                interpolation
            )
        )
