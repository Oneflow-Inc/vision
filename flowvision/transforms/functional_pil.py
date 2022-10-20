"""
Modified from https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional_pil.py
"""
import numbers
from typing import Any, List, Sequence
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

import oneflow as flow

try:
    import accimage
except ImportError:
    accimage = None


def _is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _get_image_size(img: Any) -> List[int]:
    if _is_pil_image(img):
        return img.size
    raise TypeError("Unexpected type {}".format(type(img)))


def _get_image_num_channels(img: Any) -> int:
    if _is_pil_image(img):
        return 1 if img.mode == "L" else 3


def get_dimensions(img: Any) -> List[int]:
    if _is_pil_image(img):
        if hasattr(img, "getbands"):
            channels = len(img.getbands())
        else:
            channels = img.channels
        width, height = img.size
        return [channels, height, width]
    raise TypeError(f"Unexpected type {type(img)}")


def hflip(img):
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    return img.transpose(Image.FLIP_LEFT_RIGHT)


def vflip(img):
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    return img.transpose(Image.FLIP_TOP_BOTTOM)


def adjust_brightness(img: Image.Image, brightness_factor: float) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img: Image.Image, contrast_factor: float) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img: Image.Image, saturation_factor: float) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img: Image.Image, hue_factor: float) -> Image.Image:
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError("hue_factor ({}) is not in [-0.5, 0.5].".format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img

    h, s, v = img.convert("HSV").split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")

    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img


def pad(img, padding, fill=0, padding_mode="constant"):
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, list):
        padding = tuple(padding)

    if isinstance(padding, tuple) and len(padding) not in [1, 2, 4]:
        raise ValueError(
            "Padding must be an int or a 1, 2, or 4 element tuple, not a "
            + "{} element tuple".format(len(padding))
        )

    if isinstance(padding, tuple) and len(padding) == 1:
        # Compatibility with `functional_tensor.pad`
        padding = padding[0]

    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError(
            "Padding mode should be either constant, edge, reflect or symmetric"
        )

    if padding_mode == "constant":
        opts = _parse_fill(fill, img, name="fill")
        if img.mode == "P":
            palette = img.getpalette()
            image = ImageOps.expand(img, border=padding, **opts)
            image.putpalette(palette)
            return image

        return ImageOps.expand(img, border=padding, **opts)
    else:
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        if isinstance(padding, tuple) and len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        if isinstance(padding, tuple) and len(padding) == 4:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]

        p = [pad_left, pad_top, pad_right, pad_bottom]
        cropping = -np.minimum(p, 0)

        if cropping.any():
            crop_left, crop_top, crop_right, crop_bottom = cropping
            img = img.crop(
                (crop_left, crop_top, img.width - crop_right, img.height - crop_bottom)
            )

        pad_left, pad_top, pad_right, pad_bottom = np.maximum(p, 0)

        if img.mode == "P":
            palette = img.getpalette()
            img = np.asarray(img)
            img = np.pad(
                img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode
            )
            img = Image.fromarray(img)
            img.putpalette(palette)
            return img

        img = np.asarray(img)
        # RGB image
        if len(img.shape) == 3:
            img = np.pad(
                img,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                padding_mode,
            )
        # Grayscale image
        if len(img.shape) == 2:
            img = np.pad(
                img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode
            )

        return Image.fromarray(img)


def crop(img: Image.Image, top: int, left: int, height: int, width: int) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    return img.crop((left, top, left + width, top + height))


def resize(img, size, interpolation=Image.BILINEAR):
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))
    if not (
        isinstance(size, int) or (isinstance(size, Sequence) and len(size) in (1, 2))
    ):
        raise TypeError("Got inappropriate size arg: {}".format(size))

    if isinstance(size, int) or len(size) == 1:
        if isinstance(size, Sequence):
            size = size[0]
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def _parse_fill(fill, img, name="fillcolor"):
    # Process fill color for affine transforms
    num_bands = len(img.getbands())
    if fill is None:
        fill = 0
    if isinstance(fill, (int, float)) and num_bands > 1:
        fill = tuple([fill] * num_bands)
    if isinstance(fill, (list, tuple)):
        if len(fill) != num_bands:
            msg = (
                "The number of elements in 'fill' does not match the number of "
                "bands of the image ({} != {})"
            )
            raise ValueError(msg.format(len(fill), num_bands))

        fill = tuple(fill)

    return {name: fill}


def rotate(img, angle, interpolation=0, expand=False, center=None, fill=None):
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    opts = _parse_fill(fill, img)
    return img.rotate(angle, interpolation, expand, center, **opts)


def to_grayscale(img: Image.Image, num_output_channels: int) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    if num_output_channels == 1:
        img = img.convert("L")
    elif num_output_channels == 3:
        img = img.convert("L")
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, "RGB")
    else:
        raise ValueError("num_output_channels should be either 1 or 3")

    return
