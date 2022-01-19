from flowvision import layers
from flowvision import models
from flowvision import datasets
from flowvision import transforms
from flowvision import loss
from flowvision import scheduler
from flowvision import data

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass

_image_backend = "PIL"


def set_image_backend(backend):
    """
    Specifies the package used to load images.
    Args:
        backend (string): Name of the image backend. one of {'PIL', 'accimage'}.
            The :mod:`accimage` package uses the Intel IPP library. It is
            generally faster than PIL, but does not support as many operations.
    """
    global _image_backend
    if backend not in ["PIL", "accimage"]:
        raise ValueError(
            "Invalid backend '{}'. Options are 'PIL' and 'accimage'".format(backend)
        )
    _image_backend = backend


def get_image_backend():
    """
    Gets the name of the package used to load images
    """
    return _image_backend
