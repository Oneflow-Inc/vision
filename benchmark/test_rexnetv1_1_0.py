def run_rexnetv1_1_0():

    import oneflow as flow
    import numpy as np
    from flowvision.models.rexnet import rexnetv1_1_0

    model = rexnetv1_1_0().to("cuda")
    input_shape = [16, 3, 224, 224]

    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    input_shape = np.ones(input_shape).astype(np.float32)
    x = flow.tensor(input_shape, requires_grad=False).to("cuda")

    y = model(x)
    if isinstance(y, tuple):
        y = y[0]
    y.sum().backward()
    optimizer.zero_grad()
    optimizer.step()


def test_rexnetv1_1_0(benchmark):
    benchmark(run_rexnetv1_1_0)
