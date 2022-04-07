from benchmark import *
from flowvision.models.pvt import pvt_tiny


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_pvt_tiny_batch_size1(benchmark, net=pvt_tiny, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_pvt_tiny_batch_size2(benchmark, net=pvt_tiny, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_pvt_tiny_batch_size4(benchmark, net=pvt_tiny, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_pvt_tiny_batch_size8(benchmark, net=pvt_tiny, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_pvt_tiny_batch_size16(benchmark, net=pvt_tiny, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
from benchmark import *
from flowvision.models.pvt import pvt_small


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_pvt_small_batch_size1(benchmark, net=pvt_small, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_pvt_small_batch_size2(benchmark, net=pvt_small, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_pvt_small_batch_size4(benchmark, net=pvt_small, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_pvt_small_batch_size8(benchmark, net=pvt_small, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_pvt_small_batch_size16(benchmark, net=pvt_small, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
