import itertools
import numpy as np
import os
import unittest
from collections import OrderedDict
from collections.abc import Iterable
from PIL import Image

import oneflow as flow
import flowvision.transforms as transforms
import flowvision.transforms.functional as F
import flowvision.transforms.functional_tensor as F_t


def GenCartesianProduct(sets):
    assert isinstance(sets, Iterable)
    for set in sets:
        assert isinstance(set, Iterable)
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            if "gpu" in set:
                set.remove("gpu")
            if "cuda" in set:
                set.remove("cuda")
    return itertools.product(*sets)


def GenArgList(arg_dict):
    assert isinstance(arg_dict, OrderedDict)
    assert all([isinstance(x, list) for x in arg_dict.values()])
    sets = [arg_set for (_, arg_set) in arg_dict.items()]
    return GenCartesianProduct(sets)


def _test_adjust_brightness(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    x_pil = Image.fromarray(x_np, mode="RGB")

    # test 0
    y_pil = F.adjust_brightness(x_pil, 1)
    y_np = np.array(y_pil)
    self.assertTrue(np.allclose(y_np, x_np))

    # test 1
    y_pil = F.adjust_brightness(x_pil, 0.5)
    y_np = np.array(y_pil)
    y_ans = [0, 2, 6, 27, 67, 113, 18, 4, 117, 45, 127, 0]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    self.assertTrue(np.allclose(y_np, y_ans))

    # test 2
    y_pil = F.adjust_brightness(x_pil, 2)
    y_np = np.array(y_pil)
    y_ans = [0, 10, 26, 108, 255, 255, 74, 16, 255, 180, 255, 2]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    self.assertTrue(np.allclose(y_np, y_ans))


def _test_adjust_contrast(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    x_pil = Image.fromarray(x_np, mode="RGB")

    # test 0
    y_pil = F.adjust_contrast(x_pil, 1)
    y_np = np.array(y_pil)
    self.assertTrue(np.allclose(y_np, x_np))

    # test 1
    y_pil = F.adjust_contrast(x_pil, 0.5)
    y_np = np.array(y_pil)
    y_ans = [43, 45, 49, 70, 110, 156, 61, 47, 160, 88, 170, 43]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    self.assertTrue(np.allclose(y_np, y_ans))

    # test 2
    y_pil = F.adjust_contrast(x_pil, 2)
    y_np = np.array(y_pil)
    y_ans = [0, 0, 0, 22, 184, 255, 0, 0, 255, 94, 255, 0]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    self.assertTrue(np.allclose(y_np, y_ans))


def _test_adjust_saturation(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    x_pil = Image.fromarray(x_np, mode="RGB")

    # test 0
    y_pil = F.adjust_saturation(x_pil, 1)
    y_np = np.array(y_pil)
    self.assertTrue(np.allclose(y_np, x_np))

    # test 1
    y_pil = F.adjust_saturation(x_pil, 0.5)
    y_np = np.array(y_pil)
    y_ans = [2, 4, 8, 87, 128, 173, 39, 25, 138, 133, 216, 89]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    self.assertTrue(np.allclose(y_np, y_ans))

    # test 2
    y_pil = F.adjust_saturation(x_pil, 2)
    y_np = np.array(y_pil)
    y_ans = [0, 6, 22, 0, 149, 255, 32, 0, 255, 3, 255, 0]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    self.assertTrue(np.allclose(y_np, y_ans))


def _test_adjust_hue(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    x_pil = Image.fromarray(x_np, mode="RGB")

    with self.assertRaises(ValueError):
        F.adjust_hue(x_pil, -0.7)
        F.adjust_hue(x_pil, 1)

    # test 0
    y_pil = F.adjust_hue(x_pil, 0)
    y_np = np.array(y_pil)
    y_ans = [0, 5, 13, 54, 139, 226, 35, 8, 234, 91, 255, 1]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    self.assertTrue(np.allclose(y_np, y_ans))

    # test 1
    y_pil = F.adjust_hue(x_pil, 0.25)
    y_np = np.array(y_pil)
    y_ans = [13, 0, 12, 224, 54, 226, 234, 8, 99, 1, 222, 255]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    self.assertTrue(np.allclose(y_np, y_ans))

    # test 2
    y_pil = F.adjust_hue(x_pil, -0.25)
    y_np = np.array(y_pil)
    y_ans = [0, 13, 2, 54, 226, 58, 8, 234, 152, 255, 43, 1]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    self.assertTrue(np.allclose(y_np, y_ans))


def _test_randomness(fn, trans, seed, p):
    flow.manual_seed(seed)
    img = transforms.ToPILImage()(flow.rand(3, 16, 18))
    expected_transformed_img = fn(img)
    randomly_transformer_img = trans(p=p)(img)
    if p == 0:
        assert randomly_transformer_img == img
    elif p == 1:
        assert randomly_transformer_img == expected_transformed_img

    trans().__repr__()


class TestTransform(unittest.TestCase):
    def test_randomness(self):
        arg_dict = OrderedDict()
        arg_dict["trans_pair"] = [
            (F.vflip, transforms.RandomVerticalFlip),
            (F.hflip, transforms.RandomHorizontalFlip),
        ]
        arg_dict["seed"] = [*range(10)]
        arg_dict["p"] = [0, 1]
        for arg in GenArgList(arg_dict):
            _test_randomness(*arg[0], *arg[1:])

    def test_photometric_distort(self):
        _test_adjust_brightness(self)
        _test_adjust_contrast(self)
        _test_adjust_saturation(self)
        _test_adjust_hue(self)


if __name__ == "__main__":
    unittest.main()
