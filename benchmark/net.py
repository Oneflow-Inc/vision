import oneflow as flow
import numpy as np

learning_rate = 0.01
mom = 0.9


def transfrom_input_shape(ls):
    input_shape = np.ones(ls).astype(np.float32)
    return flow.tensor(input_shape, requires_grad=False).to('cuda')


def run_net(model, input_shape):
    model = model().to('cuda')
    optimizer = flow.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=mom)
    x = transfrom_input_shape(input_shape)
    model(x).sum().backward()
    optimizer.zero_grad()
    optimizer.step()
