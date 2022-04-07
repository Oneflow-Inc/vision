import numpy as np
import oneflow as flow
import gc
import functools
import sys
import json


def run(model, x, optimizer):
    y = model(x)
    if isinstance(y, tuple):
        y[0].sum().backward()
    else:
        y.sum().backward()
    optimizer.zero_grad()
    optimizer.step()
    flow.comm.barrier()


def fetch_args(net, input_shape):
    model = net().to("cuda")
    optimizer = flow.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    data = np.ones(input_shape).astype(np.float32)
    x = flow.tensor(data, requires_grad=False).to("cuda")
    return model, x, optimizer


def gc_wrapper(func):
    def inner(benchmark):
        gc.collect()
        ret = func(benchmark)
        return ret
    return inner

def compare_args(args):
    def decorator(func):
        func_name = func.__name__
        file_name = sys._getframe().f_back.f_code.co_filename
        print('oneflow-benchmark-function::', end='')
        print({
            'func_name': func_name,
            'file_name': file_name,
            'args': args
            })
        @gc_wrapper
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator