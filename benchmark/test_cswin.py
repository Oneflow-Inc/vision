from benchmark import *
import oneflow_benchmark
from flowvision.models.cswin import cswin_tiny_224


@oneflow_benchmark.ci_settings(compare={"median": "5%"}, retry={"stddev":3, "times": 3})
def test_cswin_tiny_224_batch_size1(benchmark, net=cswin_tiny_224, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"}, retry={"stddev":3, "times": 3})
def test_cswin_tiny_224_batch_size2(benchmark, net=cswin_tiny_224, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"}, retry={"stddev":3, "times": 3})
def test_cswin_tiny_224_batch_size4(benchmark, net=cswin_tiny_224, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"}, retry={"stddev":3, "times": 3})
def test_cswin_tiny_224_batch_size8(benchmark, net=cswin_tiny_224, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"}, retry={"stddev":3, "times": 3})
def test_cswin_tiny_224_batch_size16(benchmark, net=cswin_tiny_224, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
