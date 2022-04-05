from benchmark import *
from flowvision.models.poolformer import poolformer_s12


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_s12_batch_size1(benchmark, net=poolformer_s12, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_s12_batch_size2(benchmark, net=poolformer_s12, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_s12_batch_size4(benchmark, net=poolformer_s12, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_s12_batch_size8(benchmark, net=poolformer_s12, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_s12_batch_size16(benchmark, net=poolformer_s12, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
from benchmark import *
from flowvision.models.poolformer import poolformer_s24


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_s24_batch_size1(benchmark, net=poolformer_s24, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_s24_batch_size2(benchmark, net=poolformer_s24, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_s24_batch_size4(benchmark, net=poolformer_s24, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_s24_batch_size8(benchmark, net=poolformer_s24, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_s24_batch_size16(benchmark, net=poolformer_s24, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
from benchmark import *
from flowvision.models.poolformer import poolformer_s36


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_s36_batch_size1(benchmark, net=poolformer_s36, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_s36_batch_size2(benchmark, net=poolformer_s36, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_s36_batch_size4(benchmark, net=poolformer_s36, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_s36_batch_size8(benchmark, net=poolformer_s36, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_s36_batch_size16(benchmark, net=poolformer_s36, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
from benchmark import *
from flowvision.models.poolformer import poolformer_m36


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_m36_batch_size1(benchmark, net=poolformer_m36, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_m36_batch_size2(benchmark, net=poolformer_m36, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_m36_batch_size4(benchmark, net=poolformer_m36, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_m36_batch_size8(benchmark, net=poolformer_m36, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_m36_batch_size16(benchmark, net=poolformer_m36, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
from benchmark import *
from flowvision.models.poolformer import poolformer_m48


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_m48_batch_size1(benchmark, net=poolformer_m48, input_shape=[1, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_m48_batch_size2(benchmark, net=poolformer_m48, input_shape=[2, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_m48_batch_size4(benchmark, net=poolformer_m48, input_shape=[4, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_m48_batch_size8(benchmark, net=poolformer_m48, input_shape=[8, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)


@compare_args(["--benchmark-compare-fail=median:5%"])
def test_poolformer_m48_batch_size16(benchmark, net=poolformer_m48, input_shape=[16, 3, 224, 224]):
    model, x, optimizer = fetch_args(net, input_shape)
    benchmark(run, model, x, optimizer)
