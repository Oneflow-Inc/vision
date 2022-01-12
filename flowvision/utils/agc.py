"""
Modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py
"""

import oneflow as flow

# TODO: only supported norm_type=1.0 now
def unitwise_norm(x, norm_type=2.0):
    if x.ndim < 1:
        return x.norm(norm_type)
    else:
        # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)


# TODO: Switch mul and clamp to inplace version
# TODO: Add test
def adaptive_clip_grad(parameters, clip_factor=0.01, eps=1e-3, norm_type=2.0):
    if isinstance(parameters, flow.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is None:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()
        max_norm = (
            unitwise_norm(p_data, norm_type=norm_type).clamp(min=eps).mul(clip_factor)
        )
        grad_norm = unitwise_norm(g_data, norm_type=norm_type)
        clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
        new_grads = flow.where(grad_norm < max_norm, g_data, clipped_grad)
        p.grad.detach().copy_(new_grads)
