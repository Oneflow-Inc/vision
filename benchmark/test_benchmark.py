from benchmark.resnest import *
from benchmark.alexnet import *
from benchmark.convnext import *
from benchmark.crossformer import *
from benchmark.cswin import *
from benchmark.densenet import *
from benchmark.ghostnet import *
from benchmark.googlenet import *
from benchmark.inception import *
from benchmark.mlp_mixer import *
from benchmark.mnasnet import *
from benchmark.mobilenet import *
from benchmark.net import *
from benchmark.poolformer import *
from benchmark.pvt import *
from benchmark.res2net import *
from benchmark.res_mlp import *
from benchmark.resnest import *
from benchmark.resnet import *
from benchmark.rexnet import *
from benchmark.shufflenet import *
from benchmark.squeezenet import *
from benchmark.swin_transformer import *
from benchmark.test_benchmark import *
from benchmark.uniformer import *
from benchmark.vision_transformer import *


def test_resnest50(benchmark):
    benchmark(run_resnest50)


def test_resnest101(benchmark):
    benchmark(run_resnest101)


# def test_resnest200(benchmark):
#    benchmark(run_resnest200)


# def test_resnest269(benchmark):
#    benchmark(run_resnest269)


def test_convnext(benchmark):
    benchmark(run_convnext)


def test_mlp_mixer(benchmark):
    benchmark(run_mlp_mixer)


def test_cswin(benchmark):
    benchmark(run_cswin)


def test_crossformer(benchmark):
    benchmark(run_crossformer)


def test_poolformer_s12(benchmark):
    benchmark(run_poolformer_s12)


def test_poolformer_s24(benchmark):
    benchmark(run_poolformer_s24)


def test_poolformer_s36(benchmark):
    benchmark(run_poolformer_s36)


def test_poolformer_m36(benchmark):
    benchmark(run_poolformer_m36)


def test_poolformer_m48(benchmark):
    benchmark(run_poolformer_m48)


def test_res_mlp(benchmark):
    benchmark(run_res_mlp)


def test_uniformer_base(benchmark):
    benchmark(run_uniformer_base)


def test_uniformer_base_ls(benchmark):
    benchmark(run_uniformer_base_ls)


def test_uniformer_small(benchmark):
    benchmark(run_uniformer_small)


def test_uniformer_small_plus(benchmark):
    benchmark(run_uniformer_small_plus)


def test_pvt_tiny(benchmark):
    benchmark(run_pvt_tiny)


def test_pvt_small(benchmark):
    benchmark(run_pvt_small)


def test_vit_tiny_patch(benchmark):
    benchmark(run_vit_tiny_patch)


def test_vit_small_patch(benchmark):
    benchmark(run_vit_small_patch)


def test_vit_base_patch(benchmark):
    benchmark(run_vit_base_patch)


def test_swin_tiny_patch(benchmark):
    benchmark(run_swin_tiny_patch)


def test_swin_small_patch(benchmark):
    benchmark(run_swin_small_patch)


def test_swin_base_patch(benchmark):
    benchmark(run_swin_base_patch)


def test_res2net(benchmark):
    benchmark(run_res2net)


def test_squeezenet(benchmark):
    benchmark(run_squeezenet)


def test_shufflenet(benchmark):
    benchmark(run_shufflenet)


def test_rexnet(benchmark):
    benchmark(run_rexnet)


def test_rexnet_lite(benchmark):
    benchmark(run_rexnet_lite)


def test_mobilenet_v3(benchmark):
    benchmark(run_mobilenet_v3)


def test_mobilenet_v2(benchmark):
    benchmark(run_mobilenet_v2)


def test_mnasnet(benchmark):
    benchmark(run_mnasnet)


def test_inception_v3(benchmark):
    benchmark(run_inception_v3)


def test_googlenet(benchmark):
    benchmark(run_googlenet)


def test_ghostnet(benchmark):
    benchmark(run_ghostnet)


def test_densenet(benchmark):
    benchmark(run_densenet)


def test_resnet50(benchmark):
    benchmark(run_resnet50)


def test_resnet50_32x4d(benchmark):
    benchmark(run_resnet50_32x4d)


def test_resnet50_2(benchmark):
    benchmark(run_resnet50_2)


def test_alexnet(benchmark):
    benchmark(run_alexnet)
