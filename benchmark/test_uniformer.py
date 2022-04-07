from benchmark import *
from flowvision.models.uniformer import uniformer_base


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_base_batch_size1(benchmark, net=uniformer_base, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_base_batch_size2(benchmark, net=uniformer_base, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_base_batch_size4(benchmark, net=uniformer_base, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_base_batch_size8(benchmark, net=uniformer_base, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_base_batch_size16(benchmark, net=uniformer_base, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
from benchmark import *
from flowvision.models.uniformer import uniformer_base_ls


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_base_ls_batch_size1(benchmark, net=uniformer_base_ls, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_base_ls_batch_size2(benchmark, net=uniformer_base_ls, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_base_ls_batch_size4(benchmark, net=uniformer_base_ls, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_base_ls_batch_size8(benchmark, net=uniformer_base_ls, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_base_ls_batch_size16(benchmark, net=uniformer_base_ls, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
from benchmark import *
from flowvision.models.uniformer import uniformer_small


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_small_batch_size1(benchmark, net=uniformer_small, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_small_batch_size2(benchmark, net=uniformer_small, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_small_batch_size4(benchmark, net=uniformer_small, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_small_batch_size8(benchmark, net=uniformer_small, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_small_batch_size16(benchmark, net=uniformer_small, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
from benchmark import *
from flowvision.models.uniformer import uniformer_small_plus


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_small_plus_batch_size1(benchmark, net=uniformer_small_plus, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_small_plus_batch_size2(benchmark, net=uniformer_small_plus, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_small_plus_batch_size4(benchmark, net=uniformer_small_plus, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_small_plus_batch_size8(benchmark, net=uniformer_small_plus, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%", "--benchmark-min-rounds=200", "--benchmark-max-time=2"])
def test_uniformer_small_plus_batch_size16(benchmark, net=uniformer_small_plus, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
