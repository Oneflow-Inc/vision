import oneflow as flow
import oneflow.nn as nn
from .registry import ModelCreator
from .utils import load_state_dict_from_url

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                ResidualAdd(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


@ModelCreator.register_model
def convmixer_1536_20(pretrained=False, **kwargs):
    model = ConvMixer(1536, 20, kernel_size=9, patch_size=7, n_classes=1000)
    model.default_cfg = _cfg
    return model


@ModelCreator.register_model
def convmixer_768_32(pretrained=False, **kwargs):
    model = ConvMixer(768, 32, kernel_size=7, patch_size=7, n_classes=1000)
    model.default_cfg = _cfg
    return model

    