#!/usr/bin/env bash

set -uxo pipefail

rc=0
trap 'rc=$?' ERR

# cd $ONEFLOW_MODELS_DIR
COMPARE_SCRIPT_PATH=./compare_speed_with_pytorch.py
MODELS_ROOT=../../flowvision/models
TIMES=100

function check_relative_speed {
  awk -F'[:(]' -v threshold=$1 'BEGIN { ret=2 } /Relative speed/{ if ($2 >= threshold) { printf "✔️ "; ret=0 } else { printf "❌ "; ret=1 }} {print $0} END { exit ret }'
}

function write_to_file_and_print {
  tee -a result
  printf "\n" >> result
}


# single device tests
python3 $COMPARE_SCRIPT_PATH $MODELS_ROOT/squeezenet.py squeezenet1_0 16x3x224x224 --no-show-memory --times $TIMES | check_relative_speed 1.01 | write_to_file_and_print
python3 $COMPARE_SCRIPT_PATH $MODELS_ROOT/shufflenet_v2.py shufflenet_v2_x0_5 16x3x224x224 --no-show-memory --times $TIMES | check_relative_speed 1.01 | write_to_file_and_print
python3 $COMPARE_SCRIPT_PATH $MODELS_ROOT/rexnet.py rexnetv1_1_0 16x3x224x224 --no-show-memory --times $TIMES | check_relative_speed 1.01 | write_to_file_and_print
python3 $COMPARE_SCRIPT_PATH $MODELS_ROOT/rexnet_lite.py rexnet_lite_1_0 16x3x224x224 --no-show-memory --times $TIMES | check_relative_speed 1.01 | write_to_file_and_print
python3 $COMPARE_SCRIPT_PATH $MODELS_ROOT/mobilenet_v3.py mobilenet_v3_large 16x3x224x224 --no-show-memory --times $TIMES | check_relative_speed 1.01 | write_to_file_and_print
python3 $COMPARE_SCRIPT_PATH $MODELS_ROOT/mobilenet_v2.py mobilenet_v2 16x3x224x224 --no-show-memory --times $TIMES | check_relative_speed 1.01 | write_to_file_and_print
python3 $COMPARE_SCRIPT_PATH $MODELS_ROOT/mnasnet.py mnasnet0_5 16x3x224x224 --no-show-memory --times $TIMES | check_relative_speed 1.01 | write_to_file_and_print
python3 $COMPARE_SCRIPT_PATH $MODELS_ROOT/inception_v3.py inception_v3 16x3x299x299 --no-show-memory --times $TIMES | check_relative_speed 1.01 | write_to_file_and_print
python3 $COMPARE_SCRIPT_PATH $MODELS_ROOT/googlenet.py googlenet 16x3x224x224 --no-show-memory --times $TIMES | check_relative_speed 1.01 | write_to_file_and_print
python3 $COMPARE_SCRIPT_PATH $MODELS_ROOT/ghostnet.py ghostnet 16x3x224x224 --no-show-memory --times $TIMES | check_relative_speed 1.01 | write_to_file_and_print
python3 $COMPARE_SCRIPT_PATH $MODELS_ROOT/densenet.py densenet121 16x3x224x224 --no-show-memory --times $TIMES | check_relative_speed 1.01 | write_to_file_and_print
python3 $COMPARE_SCRIPT_PATH $MODELS_ROOT/resnet.py resnet50 16x3x224x224 --no-show-memory --times $TIMES | check_relative_speed 1.01 | write_to_file_and_print
python3 $COMPARE_SCRIPT_PATH $MODELS_ROOT/resnet.py resnext50_32x4d 16x3x224x224 --no-show-memory --times $TIMES | check_relative_speed 1.01 | write_to_file_and_print
python3 $COMPARE_SCRIPT_PATH $MODELS_ROOT/resnet.py wide_resnet50_2 8x3x224x224 --no-show-memory --times $TIMES | check_relative_speed 1.01 | write_to_file_and_print
python3 $COMPARE_SCRIPT_PATH $MODELS_ROOT/alexnet.py alexnet 16x3x224x224 --no-show-memory --times $TIMES | check_relative_speed 1.01 | write_to_file_and_print


# ddp speed tests
python3 -m oneflow.distributed.launch --nproc_per_node 2 $COMPARE_SCRIPT_PATH $MODELS_ROOT/squeezenet.py squeezenet1_0 16x3x224x224 --no-show-memory --times $TIMES --ddp | check_relative_speed 0.99 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 $COMPARE_SCRIPT_PATH $MODELS_ROOT/shufflenet_v2.py shufflenet_v2_x0_5 16x3x224x224 --no-show-memory --times $TIMES --ddp | check_relative_speed 0.99 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 $COMPARE_SCRIPT_PATH $MODELS_ROOT/rexnet.py rexnetv1_1_0 16x3x224x224 --no-show-memory --times $TIMES --ddp | check_relative_speed 0.99 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 $COMPARE_SCRIPT_PATH $MODELS_ROOT/rexnet_lite.py rexnet_lite_1_0 16x3x224x224 --no-show-memory --times $TIMES --ddp | check_relative_speed 0.99 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 $COMPARE_SCRIPT_PATH $MODELS_ROOT/mobilenet_v3.py mobilenet_v3_large 16x3x224x224 --no-show-memory --times $TIMES --ddp | check_relative_speed 0.99 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 $COMPARE_SCRIPT_PATH $MODELS_ROOT/mobilenet_v2.py mobilenet_v2 16x3x224x224 --no-show-memory --times $TIMES --ddp | check_relative_speed 0.99 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 $COMPARE_SCRIPT_PATH $MODELS_ROOT/mnasnet.py mnasnet0_5 16x3x224x224 --no-show-memory --times $TIMES --ddp | check_relative_speed 0.99 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 $COMPARE_SCRIPT_PATH $MODELS_ROOT/inception_v3.py inception_v3 16x3x299x299 --no-show-memory --times $TIMES --ddp --disable-backward | check_relative_speed 0.99 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 $COMPARE_SCRIPT_PATH $MODELS_ROOT/googlenet.py googlenet 16x3x224x224 --no-show-memory --times $TIMES --ddp --disable-backward | check_relative_speed 0.99 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 $COMPARE_SCRIPT_PATH $MODELS_ROOT/ghostnet.py ghostnet 16x3x224x224 --no-show-memory --times $TIMES --ddp | check_relative_speed 0.99 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 $COMPARE_SCRIPT_PATH $MODELS_ROOT/densenet.py densenet121 16x3x224x224 --no-show-memory --times $TIMES --ddp | check_relative_speed 0.99 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 $COMPARE_SCRIPT_PATH $MODELS_ROOT/resnet.py resnet50 16x3x224x224 --no-show-memory --times $TIMES --ddp | check_relative_speed 0.99 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 $COMPARE_SCRIPT_PATH $MODELS_ROOT/resnet.py resnext50_32x4d 16x3x224x224 --no-show-memory --times $TIMES --ddp | check_relative_speed 0.99 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 $COMPARE_SCRIPT_PATH $MODELS_ROOT/resnet.py wide_resnet50_2 8x3x224x224 --no-show-memory --times $TIMES --ddp | check_relative_speed 0.99 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 $COMPARE_SCRIPT_PATH $MODELS_ROOT/alexnet.py alexnet 16x3x224x224 --no-show-memory --times $TIMES --ddp | check_relative_speed 0.99 | write_to_file_and_print


result="GPU Name: `nvidia-smi --query-gpu=name --format=csv,noheader -i 0` \n\n `cat result`"
# escape newline for github actions: https://github.community/t/set-output-truncates-multiline-strings/16852/2
# note that we escape \n and \r to \\n and \\r (i.e. raw string "\n" and "\r") instead of %0A and %0D, 
# so that they can be correctly handled in javascript code
result="${result//'%'/'%25'}"
result="${result//$'\n'/'\\n'}"
result="${result//$'\r'/'\\r'}"

echo "::set-output name=stats::$result"

exit $rc
