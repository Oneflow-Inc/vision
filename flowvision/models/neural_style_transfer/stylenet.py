"""
Modified from https://github.com/Oneflow-Inc/models/blob/main/Vision/style_transform/fast_neural_style/neural_style/transformer_net.py
"""
from typing import Any

import oneflow as flow

from ..registry import ModelCreator
from ..utils import load_state_dict_from_url


__all__ = ["NeuralStyleTransfer", "neural_style_transfer"]


style_model_urls = {
    "sketch": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/neural_style_transfer/sketch_oneflow.tar.gz",
    "candy": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/neural_style_transfer/candy_oneflow.tar.gz",
    "mosaic": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/neural_style_transfer/mosaic_oneflow.tar.gz",
    "rain_princess": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/neural_style_transfer/rain_princess_oneflow.tar.gz",
    "udnie": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/neural_style_transfer/udnie_oneflow.tar.gz",
}


class ConvLayer(flow.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = flow.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = flow.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(flow.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = flow.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = flow.nn.InstanceNorm2d(channels, affine=True)
        self.relu = flow.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(flow.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        if self.upsample:
            self.interpolate = flow.nn.UpsamplingNearest2d(scale_factor=upsample)
        self.reflection_pad = flow.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = flow.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.interpolate(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class NeuralStyleTransfer(flow.nn.Module):
    def __init__(self):
        super(NeuralStyleTransfer, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = flow.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = flow.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = flow.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = flow.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = flow.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = flow.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        y = flow.clamp(y, 0, 255)
        return y


@ModelCreator.register_model
def neural_style_transfer(
    pretrained: bool = False,
    progress: bool = True,
    style_model: str = "sketch",
    **kwargs: Any
) -> NeuralStyleTransfer:
    """
    Constructs the Neural Style Transfer model.

    .. note::
        `Perceptual Losses for Real-Time Style Transfer and Super-Resolution <https://arxiv.org/abs/1603.08155>`_.
        The required minimum input size of the model is 256x256.
        For more details for how to use this model, users can refer to: `neural_style_transfer project <https://github.com/Oneflow-Inc/vision/tree/main/projects/neural_style_transfer>`_.

    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``
        style_model (str): Which pretrained style model to download, user can choose from [sketch, candy, mosaic, rain_princess, udnie]. Default: ``sketch``

    For example:

    .. code-block:: python

        >>> import flowvision
        >>> neural_style_transfer = flowvision.models.neural_style_transfer(pretrained=True, progress=True, style_model = "sketch")

    """
    assert (
        style_model in style_model_urls.keys()
    ), "`style_model` must choose from [sketch, candy, mosaic, rain_princess, udnie]"

    model = NeuralStyleTransfer(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            style_model_urls[style_model], progress=progress
        )
        model.load_state_dict(state_dict)
    return model
