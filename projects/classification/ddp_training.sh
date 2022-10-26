export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=4
PORT=12345
MODEL_ARCH="resnet18"  # "swin_tiny_patch4_window7_224"

python3 -m oneflow.distributed.launch --nproc_per_node $GPU_NUMS --master_port $PORT  \
        main.py --cfg configs/cnn_default_settings.yaml \
                --model_arch $MODEL_ARCH


# for debug 
# python3  main.py --cfg configs/cnn_default_settings.yaml --model_arch "resnet18"