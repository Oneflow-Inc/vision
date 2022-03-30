import numpy as np
import oneflow as flow


def run_inception_v3_batch_size16(model, data, optimizer):
    x = flow.tensor(data, requires_grad=False).to("cuda")
    y = model(x)
    if isinstance(y, tuple):
        y = y[0]
    y.sum().backward()
    optimizer.zero_grad()
    optimizer.step()


def test_inception_v3_batch_size16(benchmark):
    from flowvision.models.inception_v3 import inception_v3
    input_shape = [16, 3, 299, 299]
    model = inception_v3().to("cuda")
    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    data = np.ones(input_shape).astype(np.float32)
    benchmark(run_inception_v3_batch_size16, model, data, optimizer)
    import gc
    gc.collect()


def run_inception_v3_batch_size8(model, data, optimizer):
    x = flow.tensor(data, requires_grad=False).to("cuda")
    y = model(x)
    if isinstance(y, tuple):
        y = y[0]
    y.sum().backward()
    optimizer.zero_grad()
    optimizer.step()


def test_inception_v3_batch_size8(benchmark):
    from flowvision.models.inception_v3 import inception_v3
    input_shape = [8, 3, 299, 299]
    model = inception_v3().to("cuda")
    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    data = np.ones(input_shape).astype(np.float32)
    benchmark(run_inception_v3_batch_size8, model, data, optimizer)
    import gc
    gc.collect()


def run_inception_v3_batch_size4(model, data, optimizer):
    x = flow.tensor(data, requires_grad=False).to("cuda")
    y = model(x)
    if isinstance(y, tuple):
        y = y[0]
    y.sum().backward()
    optimizer.zero_grad()
    optimizer.step()


def test_inception_v3_batch_size4(benchmark):
    from flowvision.models.inception_v3 import inception_v3
    input_shape = [4, 3, 299, 299]
    model = inception_v3().to("cuda")
    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    data = np.ones(input_shape).astype(np.float32)
    benchmark(run_inception_v3_batch_size4, model, data, optimizer)
    import gc
    gc.collect()


def run_inception_v3_batch_size2(model, data, optimizer):
    x = flow.tensor(data, requires_grad=False).to("cuda")
    y = model(x)
    if isinstance(y, tuple):
        y = y[0]
    y.sum().backward()
    optimizer.zero_grad()
    optimizer.step()


def test_inception_v3_batch_size2(benchmark):
    from flowvision.models.inception_v3 import inception_v3
    input_shape = [2, 3, 299, 299]
    model = inception_v3().to("cuda")
    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    data = np.ones(input_shape).astype(np.float32)
    benchmark(run_inception_v3_batch_size2, model, data, optimizer)
    import gc
    gc.collect()


def run_inception_v3_batch_size1(model, data, optimizer):
    x = flow.tensor(data, requires_grad=False).to("cuda")
    y = model(x)
    if isinstance(y, tuple):
        y = y[0]
    y.sum().backward()
    optimizer.zero_grad()
    optimizer.step()


def test_inception_v3_batch_size1(benchmark):
    from flowvision.models.inception_v3 import inception_v3
    input_shape = [1, 3, 299, 299]
    model = inception_v3().to("cuda")
    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    data = np.ones(input_shape).astype(np.float32)
    benchmark(run_inception_v3_batch_size1, model, data, optimizer)
    import gc
    gc.collect()


