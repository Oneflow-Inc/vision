"""
Modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/poly_lr.py
"""

import math
import logging

import oneflow as flow

from .scheduler import Scheduler


_logger = logging.getLogger(__name__)


class PolyLRScheduler(Scheduler):
    """ Polynomial LR Scheduler w/ warmup, noise, and k-decay
    k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909
    
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
        power: float = 0.5,
        lr_min: float = 0.0,
        cycle_mul: float = 1.0,
        cycle_decay: float = 1.0,
        cycle_limit: int = 1,
        warmup_t=0,
        warmup_lr_init=0,
        warmup_prefix=False,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        k_decay=1.0,
        initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            _logger.warning(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1."
            )
        self.t_initial = t_initial
        self.power = power
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        self.k_decay = k_decay
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

            if self.cycle_mul != 1:
                i = math.floor(
                    math.log(
                        1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul
                    )
                )
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = (
                    t
                    - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
                )
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.cycle_decay ** i
            lr_max_values = [v * gamma for v in self.base_values]
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min
                    + (lr_max - self.lr_min)
                    * (1 - t_curr ** k / t_i ** k) ** self.power
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
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(
                math.floor(
                    -self.t_initial
                    * (self.cycle_mul ** cycles - 1)
                    / (1 - self.cycle_mul)
                )
            )
