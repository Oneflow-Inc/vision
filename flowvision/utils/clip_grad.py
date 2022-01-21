"""
Modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/clip_grad.py
"""

import oneflow as flow

from flowvision.utils.agc import adaptive_clip_grad


# TODO: Add flow.nn.utils.clip_grad_value_
def dispatch_clip_grad(
    parameters, value: float, mode: str = "norm", norm_type: float = 2.0
):
    """ Dispatch to gradient clipping method
    Args:
        parameters (Iterable): model parameters to clip
        value (float): clipping value/factor/norm, mode dependant
        mode (str): clipping mode, one of 'norm', 'value', 'agc'
        norm_type (float): p-norm, default 2.0
    """
    if mode == "norm":
        flow.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)
    elif mode == "value":
        flow.nn.utils.clip_grad_value_(parameters, value)
    elif mode == "agc":
        adaptive_clip_grad(parameters, value, norm_type=norm_type)
    else:
        assert False, f"Unknown clip mode ({mode})."
