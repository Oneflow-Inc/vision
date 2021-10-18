import oneflow as flow
import oneflow.nn as nn


class SE(nn.Module):
    def __init__(
        self,
        channels,
        reduction=16,
        rd_channels=None,
        act_layer=nn.ReLU,
        gate_layer=nn.Sigmoid,
        mlp_bias=False,
    ):
        super(SE, self).__init__()
        rd_channels = channels // reduction if rd_channels is None else rd_channels
        self.fc1 = nn.Conv2d(channels, rd_channels, 1, bias=mlp_bias)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, 1, bias=mlp_bias)
        self.gate = gate_layer()

    def forward(self, x):
        x_avg = self.fc2(self.act(self.fc1(x.mean((2, 3), keepdim=True))))
        x_attn = self.gate(x_avg)
        return x * x_attn
