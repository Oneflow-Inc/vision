from benchmark import *
import oneflow_benchmark
from flowvision.models.inception_v3 import inception_v3


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_inception_v3_batch_size1(
    benchmark, net=inception_v3, input_shape=[1, 3, 299, 299]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_inception_v3_batch_size2(
    benchmark, net=inception_v3, input_shape=[2, 3, 299, 299]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_inception_v3_batch_size4(
    benchmark, net=inception_v3, input_shape=[4, 3, 299, 299]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_inception_v3_batch_size8(
    benchmark, net=inception_v3, input_shape=[8, 3, 299, 299]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_inception_v3_batch_size16(
    benchmark, net=inception_v3, input_shape=[16, 3, 299, 299]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
