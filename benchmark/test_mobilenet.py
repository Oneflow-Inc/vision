from benchmark import *
from flowvision.models.mobilenet_v3 import mobilenet_v3_large


@gc_wrapper
def test_mobilenet_v3_large_batch_size1(benchmark, net=mobilenet_v3_large, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@gc_wrapper
def test_mobilenet_v3_large_batch_size2(benchmark, net=mobilenet_v3_large, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@gc_wrapper
def test_mobilenet_v3_large_batch_size4(benchmark, net=mobilenet_v3_large, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@gc_wrapper
def test_mobilenet_v3_large_batch_size8(benchmark, net=mobilenet_v3_large, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@gc_wrapper
def test_mobilenet_v3_large_batch_size16(benchmark, net=mobilenet_v3_large, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
from benchmark import *
from flowvision.models.mobilenet_v2 import mobilenet_v2


@gc_wrapper
def test_mobilenet_v2_batch_size1(benchmark, net=mobilenet_v2, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@gc_wrapper
def test_mobilenet_v2_batch_size2(benchmark, net=mobilenet_v2, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@gc_wrapper
def test_mobilenet_v2_batch_size4(benchmark, net=mobilenet_v2, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@gc_wrapper
def test_mobilenet_v2_batch_size8(benchmark, net=mobilenet_v2, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@gc_wrapper
def test_mobilenet_v2_batch_size16(benchmark, net=mobilenet_v2, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
