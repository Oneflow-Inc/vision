from benchmark import *
from flowvision.models.squeezenet import squeezenet1_0


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-min-time=2"])
def test_squeezenet1_0_batch_size1(benchmark, net=squeezenet1_0, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-min-time=2"])
def test_squeezenet1_0_batch_size2(benchmark, net=squeezenet1_0, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-min-time=2"])
def test_squeezenet1_0_batch_size4(benchmark, net=squeezenet1_0, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-min-time=2"])
def test_squeezenet1_0_batch_size8(benchmark, net=squeezenet1_0, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-min-time=2"])
def test_squeezenet1_0_batch_size16(benchmark, net=squeezenet1_0, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
