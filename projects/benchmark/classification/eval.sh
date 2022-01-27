export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

MODEL="rexnet_lite_1_0"
BATCH_SIZE=64
DATA_PATH="/dataset/imagenet/extract"
IMG_SIZE=224
NORMALIZE_MODE="imagenet_default_mean_std"
CROP_PCT=0.875
INTERPOLATION="bicubic"
NUM_WORKERS=8
DEVICE=$1

CUDA_VISIBLE_DEVICES=$DEVICE python ./projects/benchmark/classification/benchmark.py --model $MODEL \
                    --data_path $DATA_PATH \
                    --batch_size $BATCH_SIZE \
                    --img_size $IMG_SIZE \
                    --normalize_mode $NORMALIZE_MODE \
                    --crop_pct $CROP_PCT \
                    --interpolation $INTERPOLATION \
                    --num_workers $NUM_WORKERS