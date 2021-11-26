import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
from oneflow import Tensor


class DropBlock(nn.Module):
    def __init__(self, block_size: int = 7, p: float = 0.5):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.p = p

    def cal_gamma(self, x: Tensor):
        gamma = (
            self.p
            * x.shape[-1] ** 2
            / (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)
        )
        return gamma

    def forward(self, x):
        if self.training:
            gamma = self.cal_gamma(x)
            mask = flow.bernoulli(flow.ones_like(x) * gamma)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x
