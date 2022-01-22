"""
Modified from https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py
"""
import oneflow as flow
import oneflow.nn as nn
from oneflow import Tensor


def stochastic_depth(
    input: Tensor, p: float, mode: str, training: bool = True
) -> Tensor:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.
    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``
    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":  # randomly samples some data of one batch and set them zero
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:  # zeros the entire input
        size = [1] * input.ndim
    # noise = flow.empty(size, dtype=input.dtype, device=input.device)
    # TODO: add tensor.bernoulli_() method
    noise = flow.ones(size, dtype=input.dtype) * survival_rate
    # TODO: for now bernoulli only has cpu implementation in oneflow
    noise = flow.bernoulli(noise).to(input.device)
    if survival_rate > 0.0:
        noise.div(survival_rate)
    return input * noise


class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "p=" + str(self.p)
        tmpstr += ", mode=" + str(self.mode)
        tmpstr += ")"
        return tmpstr
