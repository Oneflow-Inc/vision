"""
Modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/cosine_lr.py
"""

import logging
import math
import numpy as np

import oneflow as flow

from .scheduler import Scheduler


_logger = logging.getLogger(__name__)


class CosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts borrowed from timm.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py

    Args:
        optimizer: The optimizer will be used for the training process
        t_initial: The initial number of epochs. Example, 50, 100 etc.
        t_mul: updates the SGDR schedule annealing.
        lr_min: Defaults to 1e-5. The minimum learning rate to use during the scheduling. The learning rate does not ever go below this value.
        decay_rate: When decay rate > 0 and < 1., at every restart the learning rate is decayed by new learning rate which equals lr * decay_rate. If decay_rate=0.5,
                    then in that case, the new learning rate becomes half the initial lr.
        warmup_t: Defines the number of warmup epochs.
        warmup_lr_init: The initial learning rate during warmup.

    """

    def __init__(
        self,
        optimizer: flow.optim.Optimizer,
        t_initial: int,
        t_mul: float = 1.0,
        lr_min: float = 0.0,
        decay_rate: float = 1.0,
        warmup_t=0,
        warmup_lr_init=0,
        warmup_prefix=False,
        cycle_limit=0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
        )

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            _logger.warning(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1."
            )
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [
                (v - warmup_lr_init) / self.warmup_t for v in self.base_values
            ]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.t_mul != 1:
                i = math.floor(
                    math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul)
                )
                t_i = self.t_mul ** i * self.t_initial
                t_curr = t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.decay_rate ** i
            lr_min = self.lr_min * gamma
            lr_max_values = [v * gamma for v in self.base_values]

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                lrs = [
                    lr_min
                    + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def get_cycle_length(self, cycles=0):
        if not cycles:
            cycles = self.cycle_limit
        cycles = max(1, cycles)
        if self.t_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(
                math.floor(
                    -self.t_initial * (self.t_mul ** cycles - 1) / (1 - self.t_mul)
                )
            )
