import oneflow as flow
import oneflow.nn as nn
from .registry import ModelCreator
from .utils import load_state_dict_from_url

__all__ = [
    "ConvMixer",
    "convmixer_1536_20",
    "convmixer_768_32_relu",
    "convmixer_1024_20",
]

model_urls = {
    "convmixer_768_32_relu": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ConvMixer/convmixer_768_32_ks7_p7_relu.zip",
    "convmixer_1024_20": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ConvMixer/convmixer_1024_20_ks9_p14.zip",
    "convmixer_1536_20": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ConvMixer/convmixer_1536_20_ks9_p7.zip",
}


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(
    dim, depth, kernel_size=9, patch_size=7, n_classes=1000, activation=nn.GELU
):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        activation(),
        nn.BatchNorm2d(dim),
        *[
            nn.Sequential(
                ResidualAdd(
                    nn.Sequential(
                        nn.Conv2d(
                            dim, dim, kernel_size, groups=dim, padding=kernel_size // 2
                        ),
                        activation(),
                        nn.BatchNorm2d(dim),
                    )
                ),
                nn.Conv2d(dim, dim, kernel_size=1),
                activation(),
                nn.BatchNorm2d(dim),
            )
            for i in range(depth)
        ],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


@ModelCreator.register_model
def convmixer_1536_20(pretrained: bool = False, progress: bool = True, **kwargs):
    model = ConvMixer(1536, 20, kernel_size=9, patch_size=7, n_classes=1000)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["convmixer_1536_20"],
            model_dir="./checkpoints",
            progress=progress,
        )
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def convmixer_768_32_relu(pretrained: bool = False, progress: bool = True, **kwargs):
    model = ConvMixer(
        768, 32, kernel_size=7, patch_size=7, n_classes=1000, activation=nn.ReLU
    )
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["convmixer_768_32_relu"],
            model_dir="./checkpoints",
            progress=progress,
        )
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def convmixer_1024_20(pretrained: bool = False, progress: bool = True, **kwargs):
    model = ConvMixer(1024, 20, kernel_size=9, patch_size=14, n_classes=1000)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["convmixer_1024_20"],
            model_dir="./checkpoints",
            progress=progress,
        )
        model.load_state_dict(state_dict)
    return model
