import numpy as np
import oneflow as flow
import gc

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
        ret = func(benchmark)
        gc.collect()
        return ret
    return inner

