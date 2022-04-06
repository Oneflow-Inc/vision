from benchmark import *
from flowvision.models.inception_v3 import inception_v3


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-min-time=2"])
def test_inception_v3_batch_size1(benchmark, net=inception_v3, input_shape=[1, 3, 299, 299]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-min-time=2"])
def test_inception_v3_batch_size2(benchmark, net=inception_v3, input_shape=[2, 3, 299, 299]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-min-time=2"])
def test_inception_v3_batch_size4(benchmark, net=inception_v3, input_shape=[4, 3, 299, 299]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-min-time=2"])
def test_inception_v3_batch_size8(benchmark, net=inception_v3, input_shape=[8, 3, 299, 299]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-min-time=2"])
def test_inception_v3_batch_size16(benchmark, net=inception_v3, input_shape=[16, 3, 299, 299]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
