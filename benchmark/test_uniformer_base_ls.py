def run_uniformer_base_ls():

    import oneflow as flow
    import numpy as np
    from flowvision.models.uniformer import uniformer_base_ls

    model = uniformer_base_ls().to('cuda')
    input_shape = [16, 3, 224, 224]

    learning_rate = 0.01
    mom = 0.9
    optimizer = flow.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=mom)
    input_shape = np.ones(input_shape).astype(np.float32)
    x = flow.tensor(input_shape, requires_grad=False).to('cuda')

    model(x).sum().backward()
    optimizer.zero_grad()
    optimizer.step()

def test_uniformer_base_ls(benchmark):
    benchmark(run_uniformer_base_ls)
