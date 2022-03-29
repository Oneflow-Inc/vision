def run_poolformer_m48_batch_size16():

    import oneflow as flow
    import numpy as np
    from flowvision.models.poolformer import poolformer_m48

    model = poolformer_m48().to("cuda")
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
    x.numpy()
    y.numpy()


def test_poolformer_m48_batch_size16(benchmark):
    benchmark.pedantic(run_poolformer_m48_batch_size16, rounds=50)


def run_poolformer_m48_batch_size8():

    import oneflow as flow
    import numpy as np
    from flowvision.models.poolformer import poolformer_m48

    model = poolformer_m48().to("cuda")
    input_shape = [8, 3, 224, 224]

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
    x.numpy()
    y.numpy()


def test_poolformer_m48_batch_size8(benchmark):
    benchmark.pedantic(run_poolformer_m48_batch_size8, rounds=50)


def run_poolformer_m48_batch_size4():

    import oneflow as flow
    import numpy as np
    from flowvision.models.poolformer import poolformer_m48

    model = poolformer_m48().to("cuda")
    input_shape = [4, 3, 224, 224]

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
    x.numpy()
    y.numpy()


def test_poolformer_m48_batch_size4(benchmark):
    benchmark.pedantic(run_poolformer_m48_batch_size4, rounds=50)


def run_poolformer_m48_batch_size2():

    import oneflow as flow
    import numpy as np
    from flowvision.models.poolformer import poolformer_m48

    model = poolformer_m48().to("cuda")
    input_shape = [2, 3, 224, 224]

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
    x.numpy()
    y.numpy()


def test_poolformer_m48_batch_size2(benchmark):
    benchmark.pedantic(run_poolformer_m48_batch_size2, rounds=50)


def run_poolformer_m48_batch_size1():

    import oneflow as flow
    import numpy as np
    from flowvision.models.poolformer import poolformer_m48

    model = poolformer_m48().to("cuda")
    input_shape = [1, 3, 224, 224]

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
    x.numpy()
    y.numpy()


def test_poolformer_m48_batch_size1(benchmark):
    benchmark.pedantic(run_poolformer_m48_batch_size1, rounds=50)
