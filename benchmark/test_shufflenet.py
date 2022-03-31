from benchmark import *
from flowvision.models.shufflenet_v2 import shufflenet_v2_x0_5


@gc_wrapper
def test_shufflenet_v2_x0_5_batch_size1(benchmark, net=shufflenet_v2_x0_5, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@gc_wrapper
def test_shufflenet_v2_x0_5_batch_size2(benchmark, net=shufflenet_v2_x0_5, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@gc_wrapper
def test_shufflenet_v2_x0_5_batch_size4(benchmark, net=shufflenet_v2_x0_5, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@gc_wrapper
def test_shufflenet_v2_x0_5_batch_size8(benchmark, net=shufflenet_v2_x0_5, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@gc_wrapper
def test_shufflenet_v2_x0_5_batch_size16(benchmark, net=shufflenet_v2_x0_5, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)