from benchmark import *
import oneflow_benchmark
from flowvision.models.mnasnet import mnasnet0_5


@oneflow_benchmark.ci_settings(compare={"median": "30%"})
def test_mnasnet0_5_batch_size1(benchmark, net=mnasnet0_5, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "20%"})
def test_mnasnet0_5_batch_size2(benchmark, net=mnasnet0_5, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "10%"})
def test_mnasnet0_5_batch_size4(benchmark, net=mnasnet0_5, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_mnasnet0_5_batch_size8(benchmark, net=mnasnet0_5, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_mnasnet0_5_batch_size16(benchmark, net=mnasnet0_5, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
