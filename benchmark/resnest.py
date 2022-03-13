import oneflow as flow
from flowvision.models.resnest import *
import numpy as np

learning_rate = 0.01
mom = 0.9

def input_shape(ls):
    input_shape = np.ones(ls).astype(np.float32)
    return flow.tensor(input_shape, requires_grad=False).to('cuda')

def run_resnest(model, input_shape):
    model = model().to('cuda')
    optimizer = flow.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=mom)
    x = input_shape()
    model(x).sum().backward(input_shape)
    optimizer.zero_grad()
    optimizer.step()

def run_resnest50(): return run_resnest(resnest50, [16, 3, 224, 224])
def run_resnest101(): return run_resnest(resnest101, [16, 3, 256, 256])
def run_resnest200(): return run_resnest(resnest200, [16, 3, 320, 320])
def run_resnest269(): return run_resnest(resnest269, [16, 3, 416, 416])