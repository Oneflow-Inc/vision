from benchmark import *
import oneflow_benchmark
from flowvision.models.res_mlp import resmlp_12_224


@oneflow_benchmark.ci_settings(compare_args=["--benchmark-compare-fail=median:5%"])
def test_resmlp_12_224_batch_size1(benchmark, net=resmlp_12_224, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare_args=["--benchmark-compare-fail=median:5%"])
def test_resmlp_12_224_batch_size2(benchmark, net=resmlp_12_224, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare_args=["--benchmark-compare-fail=median:5%"])
def test_resmlp_12_224_batch_size4(benchmark, net=resmlp_12_224, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare_args=["--benchmark-compare-fail=median:5%"])
def test_resmlp_12_224_batch_size8(benchmark, net=resmlp_12_224, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@oneflow_benchmark.ci_settings(compare_args=["--benchmark-compare-fail=median:5%"])
def test_resmlp_12_224_batch_size16(benchmark, net=resmlp_12_224, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
