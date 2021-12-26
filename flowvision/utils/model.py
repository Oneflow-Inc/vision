"""
Modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model.py
"""
import fnmatch

import oneflow as flow
import oneflow.nn as nn

from flowvision.layers.blocks import FrozenBatchNorm2d

def unwarp_model(model):
    return model.module if hasattr(model, "module") else model


def get_state_dict(model, unwrap_fn=unwarp_model):
    return unwarp_model(model).state_dict()


def avg_sq_ch_mean(model, input, output):
    """calculate average channel square mean of output activations
    """
    return flow.mean(output.var(dim=[0, 2, 3]) ** 2).item()


def avg_ch_var(model, input, output):
    """ calculate average channel variance of output activations
    """
    return flow.mean(output.var(dim=[0, 2, 3])).item()


class ActivationStateHook:
    """Mofidied from timm to match the oneflow backend
    Iterates through each of `model`'s modules and matches modules using unix pattern 
    matching based on `hook_fn_locs` and registers `hook_fn` to the module if there is 
    a match. 

    Arguments:
        model (nn.Module): model from which we will extract the activation stats
        hook_fn_locs (List[str]): List of `hook_fn` locations based on Unix type string 
            matching with the name of model's modules. 
        hook_fns (List[Callable]): List of hook functions to be registered at every
            module in `layer_names`.
    
    Inspiration from https://docs.fast.ai/callback.hook.html.
    Refer to https://gist.github.com/amaarora/6e56942fcb46e67ba203f3009b30d950 for an example 
    on how to plot Signal Propogation Plots using `ActivationStatsHook`.
    """
    def __init__(self, model, hook_fn_locs, hook_fns):
        self.model = model
        self.hook_fn_locs = hook_fn_locs
        self.hook_fns = hook_fns
        if len(hook_fn_locs) != len(hook_fns):
            raise ValueError("Please provide `hook_fns` for each `hook_fn_locs`, \
                their lengths are different.")
        self.stats = dict((hook_fn.__name__, []) for hook_fn in hook_fns)
        for hook_fn_loc, hook_fn in zip(hook_fn_locs, hook_fns):
            self.register_hook(hook_fn_loc, hook_fn)
    
    def _create_hook(self, hook_fn):
        def append_activation_stats(module, input, output):
            out = hook_fn(module, input, output)
            self.stats[hook_fn.__name__].append(out)
        
        return append_activation_stats
    
    def register_hook(self, hook_fn_loc, hook_fn):
        for name, module in self.model.named_modules():
            if not fnmatch.fnmatch(name, hook_fn_loc):
                continue
            module.register_forward_hook(self._create_hook(hook_fn))


def freeze_batch_norm_2d(module):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (flow.nn.Module): Any OneFlow module.
    Returns:
        flow.nn.Module: Resulting module
    """
    res = module
    if isinstance(module, (nn.BatchNorm2d)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        # FIXME: Call `running_mean.data`, raise warning: You shouldn't call `.data` for a LocalTensor
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for name, child in module.named_children():
            new_child = freeze_batch_norm_2d(child)
            if new_child is not child:
                res.add_module(name, new_child)
    return res


def unfreeze_batch_norm_2d(module):
    """
    Converts all `FrozenBatchNorm2d` layers of provided module into `BatchNorm2d`. If `module` is itself and instance
    of `FrozenBatchNorm2d`, it is converted into `BatchNorm2d` and returned. Otherwise, the module is walked
    recursively and submodules are converted in place.

    Args:
        module (flow.nn.Module): Any PyTorch module.

    Returns:
        flow.nn.Module: Resulting module
    """
    res = module
    if isinstance(module, FrozenBatchNorm2d):
        res = nn.BatchNorm2d(module.num_features)
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        # FIXME: the same as `freeze_batch_norm_2d` fn
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for name, child in module.named_children():
            new_child = unfreeze_batch_norm_2d(child)
            if new_child is not child:
                res.add_module(name, new_child)
    return res