import oneflow as flow


def build_optimizer(config, model):
    param_group = {"params": [p for p in model.parameters() if p is not None]}

    if config.train.clip_grad > 0.0:
        assert config.clip_grad == 1.0, "ONLY support grad_clipping == 1.0"
        param_group["clip_grad_max_norm"] = (1.0,)
        param_group["clip_grad_norm_type"] = (2.0,)
    
    opt_lower = config.train.optim.name.lower()
    optimizer = None
    if opt_lower == "sgd":
        optimizer = flow.optim.SGD(
            [param_group],
            lr = config.train.base_lr,
            momentum = config.train.optimizer.momentum,
            weight_decay = config.train.weight_decay
        )
    return optimizer
