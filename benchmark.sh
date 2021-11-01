set -aux

MODEL="swin_large_patch4_window12_384_in22k_to_1k"
BATCH_SIZE=32
DATA_PATH="/DATA/disk1/ImageNet/extract"
IMG_SIZE=384
NUM_WORKERS=8

python benchmark.py --model $MODEL \
                    --data_path $DATA_PATH \
                    --batch_size $BATCH_SIZE \
                    --img_size $IMG_SIZE \
                    --num_workers $NUM_WORKERS