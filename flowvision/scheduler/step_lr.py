"""
Modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/step_lr.py
"""
import math
import oneflow as flow

from .scheduler import Scheduler


class StepLRScheduler(Scheduler):
    """ Step LRScheduler
    Decays the learning rate of each parameter group by 
    decay_rate every decay_t steps.

    Args:
        optimizer: The optimizer will be used for the training process
        decay_t: Period of learning rate decay.
        decay_rate: Multiplicative factor of learning rate decay. Default: 1.0.
        warmup_t: Defines the number of warmup epochs.
        warmup_lr_init: The initial learning rate during warmup.

    """

    def __init__(
        self,
        optimizer: flow.optim.Optimizer,
        decay_t: float,
        decay_rate: float = 1.0,
        warmup_t=0,
        warmup_lr_init=0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
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

        self.decay_t = decay_t
        self.decay_rate = decay_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
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
            lrs = [
                v * (self.decay_rate ** (t // self.decay_t)) for v in self.base_values
            ]
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
