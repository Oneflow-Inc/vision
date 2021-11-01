import ctypes
from .attention import SE
from .blocks import ConvBn, ConvAct, ConvBnAct
from .regularization import DropBlock

__all__ = ["SE", "ConvBn", "ConvAct", "ConvBnAct", "DropBlock"]


def lib_path():
    import os
    import glob

    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    libs = glob.glob(os.path.join(dir_path, "*.so"))
    assert len(libs) > 0, f"no .so found in {dir_path}"
    return libs[0]


ctypes.CDLL(lib_path())
