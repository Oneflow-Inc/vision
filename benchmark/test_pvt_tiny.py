def run_pvt_tiny_batch_size16():

    import oneflow as flow
    import numpy as np
    from flowvision.models.pvt import pvt_tiny

    model = pvt_tiny().to('cuda')
    input_shape = [16, 3, 224, 224]

    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=mom)
    input_shape = np.ones(input_shape).astype(np.float32)
    for i in range(100):
        x = flow.tensor(input_shape, requires_grad=False).to('cuda')
        y = model(x)   
        if isinstance(y, tuple):
            y = y[0]
        y.sum().backward()
        optimizer.zero_grad()
        optimizer.step()
        x.numpy()
        y.numpy()

def test_pvt_tiny_batch_size16(benchmark):
    benchmark(run_pvt_tiny_batch_size16)
def run_pvt_tiny_batch_size8():

    import oneflow as flow
    import numpy as np
    from flowvision.models.pvt import pvt_tiny

    model = pvt_tiny().to('cuda')
    input_shape = [8, 3, 224, 224]

    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=mom)
    input_shape = np.ones(input_shape).astype(np.float32)
    for i in range(100):
        x = flow.tensor(input_shape, requires_grad=False).to('cuda')
        y = model(x)   
        if isinstance(y, tuple):
            y = y[0]
        y.sum().backward()
        optimizer.zero_grad()
        optimizer.step()
        x.numpy()
        y.numpy()

def test_pvt_tiny_batch_size8(benchmark):
    benchmark(run_pvt_tiny_batch_size8)
def run_pvt_tiny_batch_size4():

    import oneflow as flow
    import numpy as np
    from flowvision.models.pvt import pvt_tiny

    model = pvt_tiny().to('cuda')
    input_shape = [4, 3, 224, 224]

    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=mom)
    input_shape = np.ones(input_shape).astype(np.float32)
    for i in range(100):
        x = flow.tensor(input_shape, requires_grad=False).to('cuda')
        y = model(x)   
        if isinstance(y, tuple):
            y = y[0]
        y.sum().backward()
        optimizer.zero_grad()
        optimizer.step()
        x.numpy()
        y.numpy()

def test_pvt_tiny_batch_size4(benchmark):
    benchmark(run_pvt_tiny_batch_size4)
def run_pvt_tiny_batch_size2():

    import oneflow as flow
    import numpy as np
    from flowvision.models.pvt import pvt_tiny

    model = pvt_tiny().to('cuda')
    input_shape = [2, 3, 224, 224]

    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=mom)
    input_shape = np.ones(input_shape).astype(np.float32)
    for i in range(100):
        x = flow.tensor(input_shape, requires_grad=False).to('cuda')
        y = model(x)   
        if isinstance(y, tuple):
            y = y[0]
        y.sum().backward()
        optimizer.zero_grad()
        optimizer.step()
        x.numpy()
        y.numpy()

def test_pvt_tiny_batch_size2(benchmark):
    benchmark(run_pvt_tiny_batch_size2)
def run_pvt_tiny_batch_size1():

    import oneflow as flow
    import numpy as np
    from flowvision.models.pvt import pvt_tiny

    model = pvt_tiny().to('cuda')
    input_shape = [1, 3, 224, 224]

    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=mom)
    input_shape = np.ones(input_shape).astype(np.float32)
    for i in range(100):
        x = flow.tensor(input_shape, requires_grad=False).to('cuda')
        y = model(x)   
        if isinstance(y, tuple):
            y = y[0]
        y.sum().backward()
        optimizer.zero_grad()
        optimizer.step()
        x.numpy()
        y.numpy()

def test_pvt_tiny_batch_size1(benchmark):
    benchmark(run_pvt_tiny_batch_size1)
