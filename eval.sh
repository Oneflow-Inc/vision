python3 -m oneflow.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --eval \
                                                                                      --cfg configs/swin_base_patch4_window7_224.yaml \
                                                                                      --data-path /DATA/disk1/ImageNet/extract/ \
                                                                                      --resume file-path