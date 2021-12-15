"""
Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
"""
from typing import Any

import oneflow as flow
import oneflow.nn as nn

from .utils import load_state_dict_from_url
from .registry import ModelCreator


__all__ = ["AlexNet", "alexnet"]


model_urls = {
    "alexnet": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/AlexNet/alexnet_oneflow_model.tar.gz",
}


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        x = self.classifier(x)
        return x


@ModelCreator.register_model
def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    """
    Constructs the AlexNet model.

    .. note::
        AlexNet model architecture from the `One weird trick... <https://arxiv.org/abs/1404.5997>`_ paper.
        The required minimum input size of the model is 63x63.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> alexnet = flowvision.models.alexnet(pretrained=False, progress=True)

    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["alexnet"], model_dir="./checkpoints", progress=progress
        )
        model.load_state_dict(state_dict)
    return model
