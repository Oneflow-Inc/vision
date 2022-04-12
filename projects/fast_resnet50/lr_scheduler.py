import oneflow as flow


def build_lr_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.train.epochs * n_iter_per_epoch)
    warmup_steps = int(config.train.warmup_epochs * n_iter_per_epoch)
    lr_scheduler = flow.optim.lr_scheduler.CosineDecayLR(
        optimizer, decay_steps=num_steps
    )
    if config.warmup_epochs > 0:
        lr_scheduler = flow.optim.lr_scheduler.WarmUpLR(
            lr_scheduler,
            warmup_factor=0.01,
            warmup_iters=warmup_steps,
            warmup_method="linear"
        )
    return lr_scheduler
