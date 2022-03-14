python3 -m oneflow.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/swin_small_patch4_window7_224.yaml \
                                                                                          --data-path /DATA/disk1/ImageNet/extract/ \
                                                                                          --batch-size 32 \
                                                                                          --tag fix_ddp \
                                                                                          --resume /DATA/disk1/rentianhe/code/OneFlow-Models/large_scale_training/swin_transformer/output/swin_small_patch4_window7_224/fix_ddp/model_2