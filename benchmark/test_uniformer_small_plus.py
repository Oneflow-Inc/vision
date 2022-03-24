def run_uniformer_small_plus():

    import oneflow as flow
    import numpy as np
    from flowvision.models.uniformer import uniformer_small_plus

    model = uniformer_small_plus().to("cuda")
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


def test_uniformer_small_plus(benchmark):
    benchmark(run_uniformer_small_plus)