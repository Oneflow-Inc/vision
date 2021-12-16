## Model Zoo
Here we provide our test results of pretrained model on ImageNet2012, all tests were done using single TiTanV GPU with batch_size 64.The default `image size` equals to 224. When testing the pretrained weight under default settings, we first enlarge the image according to the `crop-size` and then use `centercrop` method to crop the image to the corresponding size.

**example**
When `crop-size` equals to 0.875, we first resize the image from `224` to `(224 / 0.875) = 256` and then use centercrop operation to get the input 224 image.

| Model Name (paper link)|    Top1 |    Top5 |Top-1(real)|Top-5(real)| #params| FLOPs| CPU latency | GPU throughput | Image size | Crop-size | Interpolation | chongce 2021/12/15
|:----------------------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-----------:|:--------:|:----------:|:---------:|:-------------:|-------------:|
| Alexnet                | 56.5220 | 79.0680 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    | 1
| Vgg11                  | 69.0200 | 88.6280 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    | 1
| Vgg13                  | 69.9280 | 89.2460 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    | 1
| Vgg16                  | 71.5920 | 90.3820 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    | 1
| Vgg19                  | 72.3760 | 90.8760 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    | 1
| Vgg11_bn               | 70.3700 | 89.8100 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    | 1
| Vgg13_bn               | 71.5860 | 90.3740 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Vgg16_bn               | 73.3600 | 91.5160 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Vgg19_bn               | 74.2180 | 91.8420 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Resnet18               | 69.7580 | 89.0780 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Resnet34               | 73.3140 | 91.4200 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Resnet50               | 76.1300 | 92.8620 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Resnet101              | 77.3740 | 93.5460 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Resnet152              | 78.3120 | 94.0460 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Resnext-50-32x4d       | 77.6180 | 93.6980 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Resnext-101-32x8d      | 79.3120 | 94.5260 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Wide resnet-50-2       | 78.4680 | 94.0860 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Wide resnet-101-2      | 78.8480 | 94.2840 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Densenet121            | 74.4340 | 91.9720 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Densenet169            | 75.6000 | 92.8060 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Densenet201            | 76.8960 | 93.3700 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Densenet161            | 77.1380 | 93.5600 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Googlenet              | 69.7780 | 89.5300 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Inception_v3           | 77.5080 | 93.6680 |         |         |         |         |             |          |   299      |  1.0      |   bilinear    |1
| Squeezenet 1.0         | 58.0920 | 80.4200 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Squeezenet 1.1         | 58.1780 | 80.6240 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Shufflenet_v2_x0_5     | 60.5520 | 81.7460 |         |         |   1.4M  |   41M   |             |          |   224      |  0.875    |   bilinear    |1
| Shufflenet_v2_x1_0     | 69.3620 | 88.3160 |         |         |   2.3M  |   146M  |             |          |   224      |  0.875    |   bilinear    |1
| Mobilenet_v2           | 71.8780 | 90.2860 |         |         |   3.5M  |   300M  |             |          |   224      |  0.875    |   bilinear    |1
| Mobilenet_v3_small     | 67.6680 | 87.4020 |         |         |   2.5M  |   56M   |             |          |   224      |  0.875    |   bilinear    |1
| Mobilenet_v3_large     | 74.0420 | 91.3400 |         |         |   5.4M  |   219M  |             |          |   224      |  0.875    |   bilinear    |1
| Mnasnet 0.5            | 67.7340 | 87.4900 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Mnasnet 1.0            | 73.4560 | 91.5100 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Ghostnet               | 73.9820 | 91.4620 |         |         |   5.2M  |   141M  |             |          |   224      |  0.875    |   bilinear    |1
| Rexnetv1_1_0           | 77.8440 | 93.9240 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Rexnetv1_1_3           | 79.4600 | 94.7500 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Rexnetv1_1_5           | 80.2540 | 95.1760 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Rexnetv1_2_0           | 81.5740 | 95.6440 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Rexnetv1_3_0           | 82.6320 | 96.2500 |         |         |         |         |             |          |   224      |  0.875    |   bilinear    |1
| Vit-b-16-384           | 84.2240 | 97.2180 |         |         |   86M   |   55.4G |             |          |   384      |  1.0      |   bilinear    | 1
| Vit-b-32-384           | 81.6700 | 96.1280 |         |         |         |         |             |          |   384      |  1.0      |   bilinear    | 1
| Vit-l-16-384           | 85.1540 | 97.3600 |         |         |   307M  |  190.7G |             |          |   384      |  1.0      |   bilinear    | 1
| Vit-l-32-384           | 81.5080 | 96.0900 |         |         |         |         |             |          |   384      |  1.0      |   bilinear    | 1
| Convmixer_768_32_relu  | 80.0760 | 94.9920 |         |         |   21.1M |         |             |          |   224      |  0.875    |   bilinear    | 1
| Convmixer_1024_20      | 77.0120 | 93.3840 |         |         |   24.4M |         |             |          |   224      |  0.875    |   bilinear    | 1
| Convmixer_1536_20      | 81.0560 | 95.6200 |         |         |   51.6M |         |             |          |   224      |  0.875    |   bilinear    | 1
| Swin_tiny_patch4_window7_224  | 81.1860 | 95.5100 |         |         |    28M  |   4.5G  |             |          | 224 |  0.875    |   bicubic     | 1 
| Swin_small_patch4_window7_224 | 83.1820 | 96.2400 |         |         |    50M  |   8.7G  |             |          | 224 |  0.875    |   bicubic     | 1 
| Swin_base_patch4_window7_224  | 83.4180 | 96.4460 |         |         |    88M  |   15.4G |             |          | 224 |  0.875    |   bicubic     | 1 
| Swin_base_patch4_window12_384 | 84.4760 | 96.8920 |         |         |    88M  |   47.1G |             |          | 384 |  1.0      |   bicubic     | 1 
| Swin_base_patch4_window7_224_in22k_to_1k   | 85.1260 | 97.4800 |         |         |    88M  |   15.4G |             |          | 224 | 0.875 | bicubic | 1 
| Swin_base_patch4_window12_384_in22k_to_1k  | 86.4300 | 98.0660 |         |         |    88M  |   47.1G |             |          | 384 | 1.0   | bicubic | 1 
| Swin_large_patch4_window7_224_in22k_to_1k  | 86.2480 | 97.8780 |         |         |    197M |   34.5G |             |          | 224 | 0.875 | bicubic | 1
| Swin_large_patch4_window12_384_in22k_to_1k | 87.1360 | 98.2320 |         |         |    197M |   103.9G|             |          | 384 | 1.0   | bicubic | 1 
| Pvt_tiny               | 75.0960 | 92.4200 |         |         | 13.2M   |   1.9G  |             |          |   224      |  0.875    |   bicubic    |1
| Pvt_small              | 79.7620 | 94.9420 |         |         | 24.5M   |   3.8G  |             |          |   224      |  0.875    |   bicubic    |1
| Pvt_medium             | 81.1960 | 95.6420 |         |         | 44.2M   |   6.7G  |             |          |   224      |  0.875    |   bicubic    |1
| Pvt_large              | 81.6940 | 95.8520 |         |         | 61.4M   |   9.8G  |             |          |   224      |  0.875    |   bicubic    |1
| Crossformer_tiny_patch4_group7_224  | 81.5220 | 95.5200 |         |         |  27.8M  |  2.9G    |             |          | 224  | 0.875 |   bicubic |1
| Crossformer_small_patch4_group7_224 | 82.4100 | 96.0440 |         |         |  30.7M  |  4.9G    |             |          | 224  | 0.875 |   bicubic |1
| Crossformer_base_patch4_group7_224  | 83.3640 | 96.5400 |         |         |  52.0M  |  9.2G    |             |          | 224  | 0.875 |   bicubic |1
| Crossformer_large_patch4_group7_224 | 83.8020 | 96.5620 |         |         |  90.0M  |  16.1G   |             |          | 224  | 0.875 |   bicubic |1
| Cswin_tiny_224         | 82.8120 | 96.3000 |         |         |    23M  |   4.3G  |             |          |   224      |  0.9      |   bicubic    |1
| Cswin_small_224        | 83.5960 | 96.5840 |         |         |    35M  |   6.9G  |             |          |   224      |  0.9      |   bicubic    |1
| Cswin_base_224         | 84.2280 | 96.9120 |         |         |    78M  |   15.0G |             |          |   224      |  0.9      |   bicubic    |1
| Cswin_large_224        | 86.5220 | 97.9920 |         |         |    173M |   31.5G |             |          |   224      |  0.9      |   bicubic    |1
| Cswin_base_384         | 85.5100 | 97.4840 |         |         |    78M  |   47G   |             |          |   384      |  0.9      |   bicubic    |1
| Cswin_large_384        | 87.4860 | 98.3460 |         |         |    173M |   96.8G |             |          |   384      |  0.9      |   bicubic    | 1
| Resmlp_12              | 76.6080 | 93.1420 |         |         |    15M  |   3.0G  |             |          |   224      |  0.9      |   bicubic    |1
| Resmlp_12_dist         | 77.9360 | 93.6400 |         |         |      |     |             |          |   224      |  0.9      |   bicubic    |1
| Resmlp_24              | 79.3580 | 94.5320 |         |         |    30M  |   6.0G  |             |          |   224      |  0.9      |   bicubic    |1
| Resmlp_24_dist         | 80.7720 | 95.2180 |         |         |      |     |             |          |   224      |  0.9      |   bicubic    |1
| Resmlp_24_dino         |         |         |         |         |      |     |             |          |   224      |  0.9      |   bicubic    |
| Resmlp_36              | 79.7420 | 94.8860 |         |         |    45M  |   8.9G  |             |          |   224      |  0.9      |   bicubic    |1
| Resmlp_36_dist         | 81.0880 | 95.5820 |         |         |      |     |             |          |   224      |  0.9      |   bicubic    |1
| ResmlpB_24             | 80.9440 | 95.0760 |         |         |      |     |             |          |   224      |  0.9      |   bicubic    |1
| ResmlpB_24_in22k       | 84.3960 | 97.1580 |         |         |      |     |             |          |   224      |  0.9      |   bicubic    |1
| ResmlpB_24_dist        | 83.6800 | 96.6740 |         |         |      |    |             |          |   224      |  0.9      |   bicubic    |1


Acc was tested on 12/16/2021. Oneflow=0.6.0.dev20211205+cu102; Flowvision=0.5.1