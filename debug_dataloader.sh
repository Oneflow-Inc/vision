
# python3  \
#         debug_dataloader.py --cfg configs/swin_tiny_patch4_window7_224.yaml \
#         --batch-size 32 \
#         --data-path /dataset/extract/ \


# python3 -m oneflow.distributed.launch --nproc_per_node 1 --master_port 12345  \
#         debug_dataloader.py --cfg configs/swin_tiny_patch4_window7_224.yaml \
#         --batch-size 32 \
#         --data-path /dataset/extract/ \


python3 debug_dataloader.py --cfg configs/swin_tiny_patch4_window7_224.yaml \
        --batch-size 32 \
        --data-path /home/guoluqiang/prof_dataset/

# python3 -m pdb debug_dataloader.py --cfg configs/swin_tiny_patch4_window7_224.yaml --batch-size 32 --data-path /home/guoluqiang/prof_dataset