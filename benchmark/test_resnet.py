from benchmark import *
from flowvision.models.resnet import resnet50


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_resnet50_batch_size1(benchmark, net=resnet50, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_resnet50_batch_size2(benchmark, net=resnet50, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_resnet50_batch_size4(benchmark, net=resnet50, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_resnet50_batch_size8(benchmark, net=resnet50, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_resnet50_batch_size16(benchmark, net=resnet50, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
from benchmark import *
from flowvision.models.resnet import resnext50_32x4d


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_resnext50_32x4d_batch_size1(benchmark, net=resnext50_32x4d, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_resnext50_32x4d_batch_size2(benchmark, net=resnext50_32x4d, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_resnext50_32x4d_batch_size4(benchmark, net=resnext50_32x4d, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_resnext50_32x4d_batch_size8(benchmark, net=resnext50_32x4d, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_resnext50_32x4d_batch_size16(benchmark, net=resnext50_32x4d, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
from benchmark import *
from flowvision.models.resnet import wide_resnet50_2


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_wide_resnet50_2_batch_size1(benchmark, net=wide_resnet50_2, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_wide_resnet50_2_batch_size2(benchmark, net=wide_resnet50_2, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_wide_resnet50_2_batch_size4(benchmark, net=wide_resnet50_2, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_wide_resnet50_2_batch_size8(benchmark, net=wide_resnet50_2, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_wide_resnet50_2_batch_size16(benchmark, net=wide_resnet50_2, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
