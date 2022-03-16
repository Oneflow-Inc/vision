def run_cswin_tiny_224():

    import oneflow as flow
    import numpy as np
    from flowvision.models.cswin import cswin_tiny_224

    model = cswin_tiny_224().to('cuda')
    input_shape = [16, 3, 224, 224]

    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=mom)
    input_shape = np.ones(input_shape).astype(np.float32)
    x = flow.tensor(input_shape, requires_grad=False).to('cuda')

    y = model(x)   
    if isinstance(y, tuple):
        y = y[0]
    y.sum().backward()
    optimizer.zero_grad()
    optimizer.step()

def test_cswin_tiny_224(benchmark):
    benchmark(run_cswin_tiny_224)
