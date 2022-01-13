"""
Modified from https://github.com/microsoft/Swin-Transformer/blob/main/data/samplers.py
"""

import oneflow as flow


class SubsetRandomSampler(flow.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in flow.randperm(len(self.indices)).tolist())

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch
