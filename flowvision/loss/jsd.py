"""OneFlow implementation of Jsd Loss
Modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/loss/jsd.py
"""
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

from .cross_entropy import LabelSmoothingCrossEntropy


class JsdCrossEntropy(nn.Module):
    """ Jensen-Shannon Divergence + Cross-Entropy Loss

    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781

    Hacked together by / Copyright 2020 Ross Wightman
    """
    def __init__(self, num_splits=3, alpha=12, smoothing=0.1):
        super().__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        if smoothing is not None and smoothing > 0.:
            self.cross_entropy_loss = LabelSmoothingCrossEntropy(smoothing)
        else:
            self.cross_entropy_loss = nn.CrossEntropyLoss()
        # TODO: switch to F.kl_div()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
    
    def __call__(self, output, target):
        split_size = output.shape[0] // self.num_splits
        assert split_size * self.num_splits == output.shape[0]
        logits_split = flow.split(output, split_size)

        # Cross-entropy is only computed on clean images
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [F.softmax(logits, dim=1) for logits in logits_split]

        # Clamp mixture distribution to avoid axploding KL divergence
        logp_mixture = flow.clamp(flow.stack(probs).mean(dim=0), 1e-7, 1).log()
        loss += self.alpha * sum([
            self.kl_div(logp_mixture, p_split) for p_split in probs
        ]) / len(probs)
        return loss