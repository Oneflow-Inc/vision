def run_inception_v3():

    import oneflow as flow
    import numpy as np
    from flowvision.models.inception_v3 import inception_v3

    model = inception_v3().to("cuda")
    input_shape = [4, 3, 299, 299]

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


def test_inception_v3(benchmark):
    benchmark(run_inception_v3)
