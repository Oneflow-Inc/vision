import unittest
from collections import OrderedDict

import math
import numpy as np
from numpy.lib.arraysetops import union1d
import oneflow as flow
from torch._C import dtype
from test_utils import GenArgDict, GenArgList
from flowvision.loss.cross_entropy import (
    SoftTargetCrossEntropy,
    LabelSmoothingCrossEntropy,
)


def _gather_np(x, dim, index):
    """
    Gathers values along an axis specified by ``dim``.

    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    Parameters
    ----------
    dim:
        The axis along which to index
    index:
        A tensor of indices of elements to gather

    Returns
    -------
    Output Tensor
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1 :]
    self_xsection_shape = x.shape[:dim] + x.shape[dim + 1 :]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError(
            "Except for dimension "
            + str(dim)
            + ", all dimensions of index and self should be the same size"
        )
    if index.dtype != np.dtype("int_"):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(x, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)


def _softmax_np(x, dim):
    return np.exp(x) / np.sum(np.exp(x), axis=dim, keepdims=True)


def _label_smooth_np(x, target, smoothing=0.1):
    confidence = 1.0 - smoothing
    probs = _softmax_np(x, dim=-1)
    logprobs = np.ma.log(probs)
    nll_loss = -_gather_np(logprobs, 1, np.expand_dims(target, axis=1))
    nll_loss = np.squeeze(nll_loss, axis=1)
    smooth_loss = -np.mean(logprobs, axis=-1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    return np.mean(loss)


def _test_label_smooth(test_case, shape, device, smoothing):
    np_predict = np.random.rand(*shape)
    # TODO: build a solid label to fit all the situation
    np_label = np.arange(0, len(shape))
    of_predict = flow.tensor(np_predict, dtype=flow.float32, device=flow.device(device))
    of_label = flow.tensor(np_label, dtype=flow.int64, device=flow.device(device))
    of_label_smooth_cross_entropy = LabelSmoothingCrossEntropy(smoothing)
    np_smooth_loss = _label_smooth_np(np_predict, np_label, smoothing)
    of_smooth_loss = of_label_smooth_cross_entropy(of_predict, of_label)
    test_case.assertTrue(
        np.allclose(np_smooth_loss, of_smooth_loss.numpy(), 1e-4, 1e-4, equal_nan=True)
    )


class TestLabelSmooth(unittest.TestCase):
    def test_label_smooth(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["smoothing"] = [0.1, 0.2, 0.3, 0.5]
        for arg in GenArgList(arg_dict):
            _test_label_smooth(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
