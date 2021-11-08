set -aux

MODEL="cswin_tiny_224"
BATCH_SIZE=64
DATA_PATH="/DATA/disk1/ImageNet/extract"
IMG_SIZE=224
NUM_WORKERS=8

python benchmark.py --model $MODEL \
                    --data_path $DATA_PATH \
                    --batch_size $BATCH_SIZE \
                    --img_size $IMG_SIZE \
                    --num_workers $NUM_WORKERS