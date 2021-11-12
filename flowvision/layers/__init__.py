import ctypes
from .attention import *
from .blocks import *
from .regularization import *
from .weight_init import *


from .build import LAYER_REGISTRY, build_layers

print(LAYER_REGISTRY)


def lib_path():
    import os
    import glob

    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    libs = glob.glob(os.path.join(dir_path, "*.so"))
    assert len(libs) > 0, f"no .so found in {dir_path}"
    return libs[0]


# ctypes.CDLL(lib_path())
