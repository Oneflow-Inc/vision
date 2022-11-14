
GPU_NUMS=1
MODEL="maskrcnn_resnet50_fpn"
DATA_PATH="/dataset/coco2017"
BATCH_SIZE=2
LR=0.005

python3 -m oneflow.distributed.launch \
    --nproc_per_node $GPU_NUMS \
    train.py \
    --data-path $DATA_PATH \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --world-size $GPU_NUMS \
    --lr $LR \
    --test-only \
    --pretrained

