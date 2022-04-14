from benchmark import *
import oneflow_benchmark
from flowvision.models.resnet import resnet50
from flowvision.models.resnet import resnext50_32x4d
from flowvision.models.resnet import wide_resnet50_2


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_resnet50_batch_size1(benchmark, net=resnet50, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_resnet50_batch_size2(benchmark, net=resnet50, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_resnet50_batch_size4(benchmark, net=resnet50, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_resnet50_batch_size8(benchmark, net=resnet50, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_resnet50_batch_size16(benchmark, net=resnet50, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_resnext50_32x4d_batch_size1(
    benchmark, net=resnext50_32x4d, input_shape=[1, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_resnext50_32x4d_batch_size2(
    benchmark, net=resnext50_32x4d, input_shape=[2, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_resnext50_32x4d_batch_size4(
    benchmark, net=resnext50_32x4d, input_shape=[4, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_resnext50_32x4d_batch_size8(
    benchmark, net=resnext50_32x4d, input_shape=[8, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_resnext50_32x4d_batch_size16(
    benchmark, net=resnext50_32x4d, input_shape=[16, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_wide_resnet50_2_batch_size1(
    benchmark, net=wide_resnet50_2, input_shape=[1, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_wide_resnet50_2_batch_size2(
    benchmark, net=wide_resnet50_2, input_shape=[2, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_wide_resnet50_2_batch_size4(
    benchmark, net=wide_resnet50_2, input_shape=[4, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_wide_resnet50_2_batch_size8(
    benchmark, net=wide_resnet50_2, input_shape=[8, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_wide_resnet50_2_batch_size16(
    benchmark, net=wide_resnet50_2, input_shape=[16, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
