import numpy as np
import oneflow as flow


def run_vit_small_patch16_224_batch_size16(model, data, optimizer):
    x = flow.tensor(data, requires_grad=False).to("cuda")
    y = model(x)
    if isinstance(y, tuple):
        y = y[0]
    y.sum().backward()
    optimizer.zero_grad()
    optimizer.step()


def test_vit_small_patch16_224_batch_size16(benchmark):
    from flowvision.models.vision_transformer import vit_small_patch16_224
    input_shape = [16, 3, 224, 224]
    model = vit_small_patch16_224().to("cuda")
    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    data = np.ones(input_shape).astype(np.float32)
    benchmark(run_vit_small_patch16_224_batch_size16, model, data, optimizer)


def run_vit_small_patch16_224_batch_size8(model, data, optimizer):
    x = flow.tensor(data, requires_grad=False).to("cuda")
    y = model(x)
    if isinstance(y, tuple):
        y = y[0]
    y.sum().backward()
    optimizer.zero_grad()
    optimizer.step()


def test_vit_small_patch16_224_batch_size8(benchmark):
    from flowvision.models.vision_transformer import vit_small_patch16_224
    input_shape = [8, 3, 224, 224]
    model = vit_small_patch16_224().to("cuda")
    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    data = np.ones(input_shape).astype(np.float32)
    benchmark(run_vit_small_patch16_224_batch_size8, model, data, optimizer)


def run_vit_small_patch16_224_batch_size4(model, data, optimizer):
    x = flow.tensor(data, requires_grad=False).to("cuda")
    y = model(x)
    if isinstance(y, tuple):
        y = y[0]
    y.sum().backward()
    optimizer.zero_grad()
    optimizer.step()


def test_vit_small_patch16_224_batch_size4(benchmark):
    from flowvision.models.vision_transformer import vit_small_patch16_224
    input_shape = [4, 3, 224, 224]
    model = vit_small_patch16_224().to("cuda")
    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    data = np.ones(input_shape).astype(np.float32)
    benchmark(run_vit_small_patch16_224_batch_size4, model, data, optimizer)


def run_vit_small_patch16_224_batch_size2(model, data, optimizer):
    x = flow.tensor(data, requires_grad=False).to("cuda")
    y = model(x)
    if isinstance(y, tuple):
        y = y[0]
    y.sum().backward()
    optimizer.zero_grad()
    optimizer.step()


def test_vit_small_patch16_224_batch_size2(benchmark):
    from flowvision.models.vision_transformer import vit_small_patch16_224
    input_shape = [2, 3, 224, 224]
    model = vit_small_patch16_224().to("cuda")
    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    data = np.ones(input_shape).astype(np.float32)
    benchmark(run_vit_small_patch16_224_batch_size2, model, data, optimizer)


def run_vit_small_patch16_224_batch_size1(model, data, optimizer):
    x = flow.tensor(data, requires_grad=False).to("cuda")
    y = model(x)
    if isinstance(y, tuple):
        y = y[0]
    y.sum().backward()
    optimizer.zero_grad()
    optimizer.step()


def test_vit_small_patch16_224_batch_size1(benchmark):
    from flowvision.models.vision_transformer import vit_small_patch16_224
    input_shape = [1, 3, 224, 224]
    model = vit_small_patch16_224().to("cuda")
    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    data = np.ones(input_shape).astype(np.float32)
    benchmark(run_vit_small_patch16_224_batch_size1, model, data, optimizer)

