from benchmark import *
import oneflow_benchmark
from flowvision.models.swin_transformer import swin_tiny_patch4_window7_224
from flowvision.models.swin_transformer import swin_small_patch4_window7_224
from flowvision.models.swin_transformer import swin_base_patch4_window7_224


@oneflow_benchmark.ci_settings(compare={"median": "3%"})
def test_swin_tiny_patch4_window7_224_batch_size1(benchmark, net=swin_tiny_patch4_window7_224, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "3%"})
def test_swin_tiny_patch4_window7_224_batch_size2(benchmark, net=swin_tiny_patch4_window7_224, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "3%"})
def test_swin_tiny_patch4_window7_224_batch_size4(benchmark, net=swin_tiny_patch4_window7_224, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "3%"})
def test_swin_tiny_patch4_window7_224_batch_size8(benchmark, net=swin_tiny_patch4_window7_224, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "3%"})
def test_swin_tiny_patch4_window7_224_batch_size16(benchmark, net=swin_tiny_patch4_window7_224, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "3%"})
def test_swin_small_patch4_window7_224_batch_size1(benchmark, net=swin_small_patch4_window7_224, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "3%"})
def test_swin_small_patch4_window7_224_batch_size2(benchmark, net=swin_small_patch4_window7_224, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "3%"})
def test_swin_small_patch4_window7_224_batch_size4(benchmark, net=swin_small_patch4_window7_224, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "3%"})
def test_swin_small_patch4_window7_224_batch_size8(benchmark, net=swin_small_patch4_window7_224, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "3%"})
def test_swin_small_patch4_window7_224_batch_size16(benchmark, net=swin_small_patch4_window7_224, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "3%"})
def test_swin_base_patch4_window7_224_batch_size1(benchmark, net=swin_base_patch4_window7_224, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "3%"})
def test_swin_base_patch4_window7_224_batch_size2(benchmark, net=swin_base_patch4_window7_224, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "3%"})
def test_swin_base_patch4_window7_224_batch_size4(benchmark, net=swin_base_patch4_window7_224, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "3%"})
def test_swin_base_patch4_window7_224_batch_size8(benchmark, net=swin_base_patch4_window7_224, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare={"median": "3%"})
def test_swin_base_patch4_window7_224_batch_size16(benchmark, net=swin_base_patch4_window7_224, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
