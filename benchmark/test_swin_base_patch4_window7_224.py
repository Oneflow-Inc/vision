def run_swin_base_patch4_window7_224_batch_size16():

    import oneflow as flow
    import numpy as np
    from flowvision.models.swin_transformer import swin_base_patch4_window7_224

    model = swin_base_patch4_window7_224().to("cuda")
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


def test_swin_base_patch4_window7_224_batch_size16(benchmark):
    benchmark.pedantic(run_swin_base_patch4_window7_224_batch_size16, iterations=50)


def run_swin_base_patch4_window7_224_batch_size8():

    import oneflow as flow
    import numpy as np
    from flowvision.models.swin_transformer import swin_base_patch4_window7_224

    model = swin_base_patch4_window7_224().to("cuda")
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


def test_swin_base_patch4_window7_224_batch_size8(benchmark):
    benchmark.pedantic(run_swin_base_patch4_window7_224_batch_size8, iterations=50)


def run_swin_base_patch4_window7_224_batch_size4():

    import oneflow as flow
    import numpy as np
    from flowvision.models.swin_transformer import swin_base_patch4_window7_224

    model = swin_base_patch4_window7_224().to("cuda")
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


def test_swin_base_patch4_window7_224_batch_size4(benchmark):
    benchmark.pedantic(run_swin_base_patch4_window7_224_batch_size4, iterations=50)


def run_swin_base_patch4_window7_224_batch_size2():

    import oneflow as flow
    import numpy as np
    from flowvision.models.swin_transformer import swin_base_patch4_window7_224

    model = swin_base_patch4_window7_224().to("cuda")
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


def test_swin_base_patch4_window7_224_batch_size2(benchmark):
    benchmark.pedantic(run_swin_base_patch4_window7_224_batch_size2, iterations=50)


def run_swin_base_patch4_window7_224_batch_size1():

    import oneflow as flow
    import numpy as np
    from flowvision.models.swin_transformer import swin_base_patch4_window7_224

    model = swin_base_patch4_window7_224().to("cuda")
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


def test_swin_base_patch4_window7_224_batch_size1(benchmark):
    benchmark.pedantic(run_swin_base_patch4_window7_224_batch_size1, iterations=50)
