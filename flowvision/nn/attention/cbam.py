import oneflow as flow
import oneflow.nn as nn
from ..blocks import ConvBnAct

class ChannelAttn(nn.Module):
    def __init__(self, in_channels, reduction=16, rd_channels=None, act_layer = nn.ReLU, gate_layer = nn.Sigmoid, mlp_bias=False):
        super(ChannelAttn, self).__init__()
        rd_channels = in_channels // reduction if rd_channels is None else rd_channels
        self.fc1 = nn.Conv2d(in_channels, rd_channels, 1, bias=mlp_bias)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, in_channels, 1, bias=mlp_bias)
        self.gate = gate_layer()
    
    def forward(self, x):
        x_avg = self.fc2(self.act(self.fc1(x.mean((2, 3), keepdim=True))))
        x_max = self.fc2(self.act(self.fc1(x.mean((2,3), keepdim=True))))
        return x * self.gate(x_avg + x_max)

class SpatialAttn(nn.Module):
    def __init__(self, kernel_size=7, gate_layer = nn.Sigmoid):
        super(SpatialAttn, self).__init__()
        self.conv = ConvBnAct(2, 1, kernel_size=kernel_size, padding=kernel_size//2, activation=None)
        self.gate = gate_layer()
    
    def forward(self, x):
        x_attn = flow.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        x_attn = self.conv(x_attn)
        return x * self.gate(x_attn)


if __name__ == "__main__":
    x = flow.randn(1, 16, 4, 4)
    att = SpatialAttn(7)
    att(x)