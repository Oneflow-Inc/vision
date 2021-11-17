export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

MODEL="rexnetv1_1_0"
BATCH_SIZE=64
DATA_PATH="/DATA/disk1/ImageNet/extract"
IMG_SIZE=224
NUM_WORKERS=8

python ./projects/classification/benchmark.py --model $MODEL \
                    --data_path $DATA_PATH \
                    --batch_size $BATCH_SIZE \
                    --img_size $IMG_SIZE \
                    --num_workers $NUM_WORKERS