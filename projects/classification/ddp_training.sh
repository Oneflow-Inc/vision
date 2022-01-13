export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=8
MASTER_PORT=12345

python3 -m oneflow.distributed.launch --nproc_per_node $GPU_NUMS --master_port $MASTER_PORT  \
        main.py --cfg configs/deit_training_scheduler.yaml \
