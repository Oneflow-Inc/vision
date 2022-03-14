
python3 -m oneflow.distributed.launch --nproc_per_node 8 --master_port 12345  \
        debug_with_real_data_ddp.py --cfg configs/swin_tiny_patch4_window7_224.yaml \
        --batch-size 32 \
        --data-path /dataset/extract/ \
