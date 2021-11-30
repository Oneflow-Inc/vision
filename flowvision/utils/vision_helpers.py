"""
Modified from https://github.com/pytorch/vision/blob/main/torchvision/utils.py
"""

from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib
import oneflow as flow
import math

irange = range


def make_grid(
    tensor: Union[flow.Tensor, List[flow.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
) -> flow.Tensor:
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (
        isinstance(tensor, flow.Tensor)
        or (
            isinstance(tensor, list) and all(isinstance(t, flow.Tensor) for t in tensor)
        )
    ):
        raise TypeError(
            "tensor or list of tensors expected, got {}".format(type(tensor))
        )

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = flow.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = flow.cat([tensor, tensor, tensor], 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = flow.cat([tensor, tensor, tensor], 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(
                range, tuple
            ), "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img = img.clamp(min=min, max=max)
            img = (img - min) / (max - min + 1e-5)
            return img

        def norm_range(t, range):
            if range is not None:
                img = norm_ip(t, range[0], range[1])
            else:
                img = norm_ip(t, float(t.min().item()), float(t.max().item()))
            return img

        if scale_each is True:
            bs = tensor.shape[0]  # loop over mini-batch dimension
            for t in irange(bs):
                tensor[t] = norm_range(tensor[t], range)
        else:
            tensor = norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = (
        flow.zeros((num_channels, height * ymaps + padding, width * xmaps + padding))
        + pad_value
    )
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid[
                :,
                y * height + padding : (y + 1) * height,
                x * width + padding : (x + 1) * width,
            ] = tensor[k]
            k = k + 1
    return grid


def save_image(
    tensor: Union[flow.Tensor, List[flow.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    format: Optional[str] = None,
) -> None:
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image

    tensor = flow.tensor(tensor.numpy())
    grid = make_grid(
        tensor,
        nrow=nrow,
        padding=padding,
        pad_value=pad_value,
        normalize=normalize,
        range=range,
        scale_each=scale_each,
    )
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = (
        grid.mul(255)
        .add(0.5)
        .clamp(0, 255)
        .permute(1, 2, 0)
        .to("cpu", flow.uint8)
        .numpy()
    )
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
