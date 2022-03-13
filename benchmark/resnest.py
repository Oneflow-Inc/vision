import oneflow as flow
from flowvision.models.resnest import *
import numpy as np

learning_rate = 0.01
mom = 0.9

def input_shape(ls):
    input_shape = np.ones(ls).astype(np.float32)
    return flow.tensor(input_shape, requires_grad=False).to('cuda')

def run_resnest50():
    model = resnest50().to('cuda')
    optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    x = input_shape([16, 3, 224, 224])
    model(x).sum().backward()
    optimizer.zero_grad()
    optimizer.step()

def run_resnest101():
    model = resnest101().to('cuda')
    optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    x = input_shape([16, 3, 256, 256])
    model(x).sum().backward()
    optimizer.zero_grad()
    optimizer.step()

def run_resnest200():
    model = resnest200().to('cuda')
    optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    x = input_shape([16, 3, 320, 320])
    model(x).sum().backward()
    optimizer.zero_grad()
    optimizer.step()

def run_resnest269():
    model = resnest269().to('cuda')
    optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    x = input_shape([16, 3, 416, 416])
    model(x).sum().backward()
    optimizer.zero_grad()
    optimizer.step()