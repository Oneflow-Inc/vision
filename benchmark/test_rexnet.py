from benchmark import *
import oneflow_benchmark
from flowvision.models.rexnet import rexnetv1_1_0
from flowvision.models.rexnet_lite import rexnet_lite_1_0


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_rexnetv1_1_0_batch_size1(benchmark, net=rexnetv1_1_0, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_rexnetv1_1_0_batch_size2(benchmark, net=rexnetv1_1_0, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_rexnetv1_1_0_batch_size4(benchmark, net=rexnetv1_1_0, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_rexnetv1_1_0_batch_size8(benchmark, net=rexnetv1_1_0, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_rexnetv1_1_0_batch_size16(benchmark, net=rexnetv1_1_0, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_rexnet_lite_1_0_batch_size1(benchmark, net=rexnet_lite_1_0, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_rexnet_lite_1_0_batch_size2(benchmark, net=rexnet_lite_1_0, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_rexnet_lite_1_0_batch_size4(benchmark, net=rexnet_lite_1_0, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_rexnet_lite_1_0_batch_size8(benchmark, net=rexnet_lite_1_0, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_rexnet_lite_1_0_batch_size16(benchmark, net=rexnet_lite_1_0, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
