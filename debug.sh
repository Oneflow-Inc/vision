# python3 -m oneflow.distributed.launch --nproc_per_node 8 --master_port 12345  \
# main.py --cfg configs/swin_small_patch4_window7_224.yaml \
#                     --data-path /DATA/disk1/ImageNet/extract/ \
#                     --batch-size 32 \
#                     --tag fix_ddp \

python3 debug.py --cfg configs/swin_tiny_patch4_window7_224.yaml \
        --batch-size 32

# python3 -m oneflow.distributed.launch --nproc_per_node 1 --master_port 12345  \
#         debug.py --cfg configs/swin_small_patch4_window7_224.yaml \
#         --batch-size 2 \
