import oneflow as flow
import oneflow.nn as nn


class SEModule(nn.Module):
    """
    A "Squeeze-and-Excitation" block that adaptively recalibrates channel-wise feature responses. This is based on
    `"Squeeze-and-Excitation Networks" <https://arxiv.org/abs/1709.01507>`_. This unit is designed to improve the representational capacity of a network by enabling it to perform dynamic channel-wise feature recalibration.

    Args:
        channels (int): Size of each input sample 
        reduction (int): Ratio that allows us tovary the capacity and computational cost of the SE model. Default: 16
        rd_channels (int or None): Number of reduced channels. If none, uses reduction to calculate
        act_layer (flow.nn.Module): A simple acting mechanism is achieved by using a relu activation 
        gate_layer (flow.nn.Module): A simple gating mechanism is achieved by using a sigmoid activation 
        mlp_bias (bool): If True, adds a learnable bias to the output. Default: False
    """

    def __init__(
        self,
        channels,
        reduction=16,
        rd_channels=None,
        act_layer=nn.ReLU,
        gate_layer=nn.Sigmoid,
        mlp_bias=False,
    ):
        super(SEModule, self).__init__()
        rd_channels = channels // reduction if rd_channels is None else rd_channels
        self.fc1 = nn.Conv2d(channels, rd_channels, 1, bias=mlp_bias)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, 1, bias=mlp_bias)
        self.gate = gate_layer()

    def forward(self, x):
        x_avg = self.fc2(self.act(self.fc1(x.mean((2, 3), keepdim=True))))
        x_attn = self.gate(x_avg)
        return x * x_attn
