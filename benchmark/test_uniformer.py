from benchmark import *
import oneflow_benchmark
from flowvision.models.uniformer import uniformer_base
from flowvision.models.uniformer import uniformer_small_plus
from flowvision.models.uniformer import uniformer_base_ls
from flowvision.models.uniformer import uniformer_small

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_base_batch_size1(
    benchmark, net=uniformer_base, input_shape=[1, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_base_batch_size2(
    benchmark, net=uniformer_base, input_shape=[2, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_base_batch_size4(
    benchmark, net=uniformer_base, input_shape=[4, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_base_batch_size8(
    benchmark, net=uniformer_base, input_shape=[8, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_base_batch_size16(
    benchmark, net=uniformer_base, input_shape=[16, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@unittest.skipUnless(os.getenv("ONEFLOW_BENCHMARK_ALL") == "1", "set ONEFLOW_BENCHMARK_ALL=1 to run this test")
@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_base_ls_batch_size1(
    benchmark, net=uniformer_base_ls, input_shape=[1, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@unittest.skipUnless(os.getenv("ONEFLOW_BENCHMARK_ALL") == "1", "set ONEFLOW_BENCHMARK_ALL=1 to run this test")
@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_base_ls_batch_size2(
    benchmark, net=uniformer_base_ls, input_shape=[2, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@unittest.skipUnless(os.getenv("ONEFLOW_BENCHMARK_ALL") == "1", "set ONEFLOW_BENCHMARK_ALL=1 to run this test")
@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_base_ls_batch_size4(
    benchmark, net=uniformer_base_ls, input_shape=[4, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@unittest.skipUnless(os.getenv("ONEFLOW_BENCHMARK_ALL") == "1", "set ONEFLOW_BENCHMARK_ALL=1 to run this test")
@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_base_ls_batch_size8(
    benchmark, net=uniformer_base_ls, input_shape=[8, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@unittest.skipUnless(os.getenv("ONEFLOW_BENCHMARK_ALL") == "1", "set ONEFLOW_BENCHMARK_ALL=1 to run this test")
@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_base_ls_batch_size16(
    benchmark, net=uniformer_base_ls, input_shape=[16, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@unittest.skipUnless(os.getenv("ONEFLOW_BENCHMARK_ALL") == "1", "set ONEFLOW_BENCHMARK_ALL=1 to run this test")
@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_small_batch_size1(
    benchmark, net=uniformer_small, input_shape=[1, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@unittest.skipUnless(os.getenv("ONEFLOW_BENCHMARK_ALL") == "1", "set ONEFLOW_BENCHMARK_ALL=1 to run this test")
@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_small_batch_size2(
    benchmark, net=uniformer_small, input_shape=[2, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@unittest.skipUnless(os.getenv("ONEFLOW_BENCHMARK_ALL") == "1", "set ONEFLOW_BENCHMARK_ALL=1 to run this test")
@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_small_batch_size4(
    benchmark, net=uniformer_small, input_shape=[4, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@unittest.skipUnless(os.getenv("ONEFLOW_BENCHMARK_ALL") == "1", "set ONEFLOW_BENCHMARK_ALL=1 to run this test")
@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_small_batch_size8(
    benchmark, net=uniformer_small, input_shape=[8, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@unittest.skipUnless(os.getenv("ONEFLOW_BENCHMARK_ALL") == "1", "set ONEFLOW_BENCHMARK_ALL=1 to run this test")
@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_small_batch_size16(
    benchmark, net=uniformer_small, input_shape=[16, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@unittest.skipUnless(os.getenv("ONEFLOW_BENCHMARK_ALL") == "1", "set ONEFLOW_BENCHMARK_ALL=1 to run this test")
@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_small_plus_batch_size1(
    benchmark, net=uniformer_small_plus, input_shape=[1, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@unittest.skipUnless(os.getenv("ONEFLOW_BENCHMARK_ALL") == "1", "set ONEFLOW_BENCHMARK_ALL=1 to run this test")
@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_small_plus_batch_size2(
    benchmark, net=uniformer_small_plus, input_shape=[2, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@unittest.skipUnless(os.getenv("ONEFLOW_BENCHMARK_ALL") == "1", "set ONEFLOW_BENCHMARK_ALL=1 to run this test")
@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_small_plus_batch_size4(
    benchmark, net=uniformer_small_plus, input_shape=[4, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@unittest.skipUnless(os.getenv("ONEFLOW_BENCHMARK_ALL") == "1", "set ONEFLOW_BENCHMARK_ALL=1 to run this test")
@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_small_plus_batch_size8(
    benchmark, net=uniformer_small_plus, input_shape=[8, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)

@unittest.skipUnless(os.getenv("ONEFLOW_BENCHMARK_ALL") == "1", "set ONEFLOW_BENCHMARK_ALL=1 to run this test")
@oneflow_benchmark.ci_settings(compare={"median": "5%"})
def test_uniformer_small_plus_batch_size16(
    benchmark, net=uniformer_small_plus, input_shape=[16, 3, 224, 224]
):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
