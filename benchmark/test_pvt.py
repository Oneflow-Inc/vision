from benchmark import *
import oneflow_benchmark
from flowvision.models.pvt import pvt_tiny
from flowvision.models.pvt import pvt_small


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_pvt_tiny_batch_size1(benchmark, net=pvt_tiny, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_pvt_tiny_batch_size2(benchmark, net=pvt_tiny, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_pvt_tiny_batch_size4(benchmark, net=pvt_tiny, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_pvt_tiny_batch_size8(benchmark, net=pvt_tiny, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_pvt_tiny_batch_size16(benchmark, net=pvt_tiny, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_pvt_small_batch_size1(benchmark, net=pvt_small, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_pvt_small_batch_size2(benchmark, net=pvt_small, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_pvt_small_batch_size4(benchmark, net=pvt_small, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_pvt_small_batch_size8(benchmark, net=pvt_small, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_pvt_small_batch_size16(
    benchmark, net=pvt_small, input_shape=[16, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
