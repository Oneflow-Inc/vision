set -aux

MODEL="model to test"
BATCH_SIZE=64
DATA_PATH="/path/to/imagenet dataset"
IMG_SIZE=224
NUM_WORKERS=8

python benchmark.py --model $MODEL \
                    --data_path $DATA_PATH \
                    --batch_size $BATCH_SIZE \
                    --img_size $IMG_SIZE \
                    --num_workers $NUM_WORKERS