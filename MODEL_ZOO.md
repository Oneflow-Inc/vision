## Model Zoo
Here we provide our test results of pretrained model on ImageNet2012. The default `image size` equals to 224. When testing the pretrained weight under default settings, we first enlarge the image according to the `crop-size` and then use `centercrop` method to crop the image to the corresponding size.

**example**
When `crop-size` equals to 0.875, we first resize the image from `224` to `(224 / 0.875) = 256` and then use centercrop operation to get the input 224 image.

| Model Name (paper link)|    Top1 |    Top5 |Top-1(real)|Top-5(real)| #params| FLOPs| CPU latency | GPU latency | Image size | Crop-size | Interpolation |  ONNX latency |
|:----------------------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-----------:|:--------:|:----------:|:---------:|:-------------:|-------------:|
| Alexnet                | 56.5220 | 79.0680 | 63.0655 | 83.6283 | 61.1M   | 770M    | 39.314ms    | 1.098ms  |   224      |  0.875    |   bilinear    | 
| Vgg11                  | 69.0200 | 88.6280 | 76.3819 | 92.1579 | 132.9M  | 7.7G    | 325.788ms   | 2.751ms  |   224      |  0.875    |   bilinear    | 
| Vgg13                  | 69.9280 | 89.2460 | 77.2253 | 92.6874 | 133.1M  | 11.4G   | 560.146ms   | 3.146ms  |   224      |  0.875    |   bilinear    | 
| Vgg16                  | 71.5920 | 90.3820 | 79.0401 | 93.6439 | 138.4M  | 15.6G   | 669.524ms   | 3.806ms  |   224      |  0.875    |   bilinear    | 
| Vgg19                  | 72.3760 | 90.8760 | 79.4863 | 93.8702 | 143.7M  | 19.8G   | 759.762ms   | 4.586ms  |   224      |  0.875    |   bilinear    | 
| Vgg11_bn               | 70.3700 | 89.8100 | 77.9384 | 93.2297 | 132.9M  | 7.8G    | 344.126ms   | 2.755ms  |   224      |  0.875    |   bilinear    | 
| Vgg13_bn               | 71.5860 | 90.3740 | 78.9974 | 93.6567 | 133.1M  | 11.5G   | 590.912ms   | 3.532ms  |   224      |  0.875    |   bilinear    |
| Vgg16_bn               | 73.3600 | 91.5160 | 80.5666 | 94.5983 | 138.4M  | 15.7G   | 699.074ms   | 4.212ms  |   224      |  0.875    |   bilinear    |
| Vgg19_bn               | 74.2180 | 91.8420 | 81.4442 | 94.7691 | 143.7M  | 19.8G   | 798.868ms   | 4.918ms  |   224      |  0.875    |   bilinear    |
| Resnet18               | 69.7580 | 89.0780 | 77.2851 | 92.7557 | 11.7M   | 1.8G    | 73.185ms    | 2.205ms  |   224      |  0.875    |   bilinear    |
| Resnet34               | 73.3140 | 91.4200 | 80.3873 | 94.4275 | 21.8M   | 3.7G    | 126.411ms   | 3.531ms  |   224      |  0.875    |   bilinear    |
| Resnet50               | 76.1300 | 92.8620 | 82.9579 | 95.4694 | 25.6M   | 4.2G    | 139.058ms   | 4.917ms  |   224      |  0.875    |   bilinear    |
| Resnet101              | 77.3740 | 93.5460 | 83.8632 | 95.8857 | 44.6M   | 7.9G    | 230.840ms   | 8.840ms  |   224      |  0.875    |   bilinear    |
| Resnet152              | 78.3120 | 94.0460 | 84.8154 | 96.2316 | 60.2M   | 11.6G   | 329.626ms   | 13.170ms |   224      |  0.875    |   bilinear    |
| Resnext-50-32x4d       | 77.6180 | 93.6980 | 83.9507 | 95.9626 | 25.0M   | 4.3G    | 166.267ms   | 6.242ms  |   224      |  0.875    |   bilinear    |
| Resnext-101-32x8d      | 79.3120 | 94.5260 | 85.1912 | 96.4558 | 88.8M   | 16.5G   | 512.762ms   | 16.629ms |   224      |  0.875    |   bilinear    |
| Wide resnet-50-2       | 78.4680 | 94.0860 | 84.4226 | 96.2572 | 68.9M   | 11.5G   | 290.218ms   | 6.399ms  |   224      |  0.875    |   bilinear    |
| Wide resnet-101-2      | 78.8480 | 94.2840 | 84.5507 | 96.3512 | 126.9M  | 22.8G   | 527.434ms   | 11.686ms |   224      |  0.875    |   bilinear    |
| Densenet121            | 74.4340 | 91.9720 | 81.4869 | 94.8331 | 8.0M    | 2.9G    | 242.094ms   | 10.337ms |   224      |  0.875    |   bilinear    |
| Densenet169            | 75.6000 | 92.8060 | 82.3281 | 95.4480 | 14.2M   | 3.4G    | 278.913ms   | 15.071ms |   224      |  0.875    |   bilinear    |
| Densenet201            | 76.8960 | 93.3700 | 83.1864 | 95.7448 | 20.0M   | 4.4G    | 358.196ms   | 18.108ms |   224      |  0.875    |   bilinear    |
| Densenet161            | 77.1380 | 93.5600 | 83.7009 | 95.8708 | 28.7M   | 7.9G    | 455.120ms   | 15.264ms |   224      |  0.875    |   bilinear    |
| Googlenet              | 69.7780 | 89.5300 | 77.7889 | 93.0803 | 6.8M    | 1.5G    | 118.888ms   | 4.334ms  |   224      |  0.875    |   bilinear    |
| Inception_v3           | 77.5080 | 93.6680 | 83.8760 | 96.0587 | 27.2M   | 5.8G    | 302.319ms   | 7.312ms  |   299      |  1.0      |   bilinear    |
| Squeezenet 1.0         | 58.0920 | 80.4200 | 65.3906 | 85.3556 | 1.25M   | 820M    | 110.260ms   | 1.591ms  |   224      |  0.875    |   bilinear    |
| Squeezenet 1.1         | 58.1780 | 80.6240 | 65.4547 | 85.4218 | 1.24M   | 350M    | 63.463ms    | 1.490ms  |   224      |  0.875    |   bilinear    |
| Shufflenet_v2_x0_5     | 60.5520 | 81.7460 | 67.6687 | 86.4018 |   1.4M  |   41M   | 21.069ms    | 5.484ms  |   224      |  0.875    |   bilinear    |
| Shufflenet_v2_x1_0     | 69.3620 | 88.3160 | 76.4695 | 91.8056 |   2.3M  |   146M  | 39.303ms    | 5.427ms  |   224      |  0.875    |   bilinear    |
| Mobilenet_v2           | 71.8780 | 90.2860 | 79.0379 | 93.4881 |   3.5M  |   300M  | 60.407ms    | 4.096ms  |   224      |  0.875    |   bilinear    |
| Mobilenet_v3_small     | 67.6680 | 87.4020 | 74.6205 | 91.0605 |   2.5M  |   56M   | 27.726ms    | 3.798ms  |   224      |  0.875    |   bilinear    |
| Mobilenet_v3_large     | 74.0420 | 91.3400 | 80.2912 | 94.1051 |   5.4M  |   219M  | 67.774ms    | 4.760ms  |   224      |  0.875    |   bilinear    |
| Mnasnet 0.5            | 67.7340 | 87.4900 | 74.9813 | 91.1736 |  2.2M   |   140M  | 47.996ms    | 3.875ms  |   224      |  0.875    |   bilinear    |
| Mnasnet 1.0            | 73.4560 | 91.5100 | 80.2571 | 94.3421 |  4.4M   |  340M   | 86.499ms    | 3.936ms  |   224      |  0.875    |   bilinear    |
| Ghostnet               | 73.9820 | 91.4620 | 80.7054 | 94.2930 |   5.2M  |   141M  | 66.179ms    | 8.330ms  |   224      |  0.875    |   bilinear    |
| Rexnetv1_1_0           | 77.8440 | 93.9240 | 84.1642 | 96.2402 |  4.8M   |  400M   | 147.623ms   | 6.493ms  |   224      |  0.875    |   bilinear    |
| Rexnetv1_1_3           | 79.4600 | 94.7500 | 85.4196 | 96.6778 |  7.6M   |  660M   | 193.127ms   | 6.697ms  |   224      |  0.875    |   bilinear    |
| Rexnetv1_1_5           | 80.2540 | 95.1760 | 86.1392 | 97.0365 |  7.6M   |  660M   | 224.214ms   | 6.620ms  |   224      |  0.875    |   bilinear    |
| Rexnetv1_2_0           | 81.5740 | 95.6440 | 86.8373 | 97.2330 |  16M    |  1.5G   | 300.420ms   | 6.736ms  |   224      |  0.875    |   bilinear    |
| Rexnetv1_3_0           | 82.6320 | 96.2500 | 87.6636 | 97.6045 |  34M    |  3.4G   | 466.162ms   | 6.993ms  |   224      |  0.875    |   bilinear    |
| Vit-b-16-384           | 84.2240 | 97.2180 | 88.4130 | 98.1703 |   86M   |   55.4G | 2817.720ms  | 18.158ms |   384      |  1.0      |   bilinear    | 
| Vit-b-32-384           | 81.6700 | 96.1280 | 87.0209 | 97.6514 |         |         | 596.873ms   | 11.205ms |   384      |  1.0      |   bilinear    | 
| Vit-l-16-384           | 85.1540 | 97.3600 | 88.4109 | 98.1895 |   307M  |  190.7G | 8099.552ms  | 56.595ms |   384      |  1.0      |   bilinear    | 
| Vit-l-32-384           | 81.5080 | 96.0900 | 85.9150 | 97.3717 |         |         | 1770.822ms  | 22.052ms |   384      |  1.0      |   bilinear    | 
| Convmixer_768_32_relu  | 80.0760 | 94.9920 | 86.0815 | 97.0045 |   21.1M |         | 3064.026ms  | 9.199ms  |   224      |  0.875    |   bilinear    | 
| Convmixer_1024_20      | 77.0120 | 93.3840 | 83.7821 | 95.8815 |   24.4M |         | 994.395ms   | 4.164ms  |   224      |  0.875    |   bilinear    | 
| Convmixer_1536_20      | 81.0560 | 95.6200 | 86.7348 | 97.2458 |   51.6M |         | 6189.601ms  | 16.245ms |   224      |  0.875    |   bilinear    | 
| Swin_tiny_patch4_window7_224  | 81.1860 | 95.5100 | 86.6430 | 97.1433 |    28M  |   4.5G  | 562.167ms   | 9.199ms  | 224 |  0.875    |   bicubic     | 
| Swin_small_patch4_window7_224 | 83.1820 | 96.2400 | 87.5718 | 97.5084 |    50M  |   8.7G  | 930.862ms   | 17.954ms | 224 |  0.875    |   bicubic     | 
| Swin_base_patch4_window7_224  | 83.4180 | 96.4460 | 87.6892 | 97.5127 |    88M  |   15.4G | 1283.961ms  | 17.944ms | 224 |  0.875    |   bicubic     | 
| Swin_base_patch4_window12_384 | 84.4760 | 96.8920 | 88.4215 | 97.8052 |    88M  |   47.1G | 4558.562ms  | 19.612ms | 384 |  1.0      |   bicubic     | 
| Swin_base_patch4_window7_224_in22k_to_1k   | 85.1260 | 97.4800 | 89.1496 | 98.3987        |    88M  |   15.4G |  1287.231ms | 16.973ms | 224 | 0.875 | bicubic | 
| Swin_base_patch4_window12_384_in22k_to_1k  | 86.4300 | 98.0660 | 89.9951 | 98.6976        |    88M  |   47.1G |  4541.201ms | 19.556ms | 384 | 1.0   | bicubic | 
| Swin_large_patch4_window7_224_in22k_to_1k  | 86.2480 | 97.8780 | 89.7111 | 98.5674        |    197M |   34.5G |  2100.791ms | 18.194ms | 224 | 0.875 | bicubic |
| Swin_large_patch4_window12_384_in22k_to_1k | 87.1360 | 98.2320 | 90.0186 | 98.6613        |    197M |   103.9G|  7264.798ms | 35.177ms | 384 | 1.0   | bicubic | 
| Pvt_tiny               | 75.0960 | 92.4200 | 82.1893 | 95.2644 | 13.2M   |   1.9G  | 257.670ms   | 7.584ms  |   224      |  0.875    |   bicubic    |
| Pvt_small              | 79.7620 | 94.9420 | 85.7698 | 96.8999 | 24.5M   |   3.8G  | 460.928ms   | 13.601ms |   224      |  0.875    |   bicubic    |
| Pvt_medium             | 81.1960 | 95.6420 | 86.6665 | 97.1668 | 44.2M   |   6.7G  | 705.326ms   | 23.855ms |   224      |  0.875    |   bicubic    |
| Pvt_large              | 81.6940 | 95.8520 | 87.0338 | 97.3888 | 61.4M   |   9.8G  | 1000.001ms  | 33.991ms |   224      |  0.875    |   bicubic    |
| Crossformer_tiny_patch4_group7_224  | 81.5220 | 95.5200 | 86.4487 | 96.9298 |  27.8M  |  2.9G    | 396.214ms   | 19.148ms | 224  | 0.875 |   bicubic |
| Crossformer_small_patch4_group7_224 | 82.4100 | 96.0440 | 87.3796 | 97.5511 |  30.7M  |  4.9G    | 570.599ms   | 14.456ms | 224  | 0.875 |   bicubic |
| Crossformer_base_patch4_group7_224  | 83.3640 | 96.5400 | 87.7020 | 97.5618 |  52.0M  |  9.2G    | 911.867ms   | 28.412ms | 224  | 0.875 |   bicubic |
| Crossformer_large_patch4_group7_224 | 83.8020 | 96.5620 | 87.9646 | 97.6066 |  90.0M  |  16.1G   | 1236.853ms  | 29.317ms | 224  | 0.875 |   bicubic |
| Cswin_tiny_224         | 82.8120 | 96.3000 | 87.7277 | 97.5788 |    23M  |   4.3G  | 652.379.ms  | 32.788ms |   224      |  0.9      |   bicubic    |
| Cswin_small_224        | 83.5960 | 96.5840 | 88.1547 | 97.6792 |    35M  |   6.9G  | 1042.833ms  | 51.576ms |   224      |  0.9      |   bicubic    |
| Cswin_base_224         | 84.2280 | 96.9120 | 88.2700 | 97.9290 |    78M  |   15.0G | 1645.062ms  | 50.770ms |   224      |  0.9      |   bicubic    |
| Cswin_large_224        | 86.5220 | 97.9920 | 89.6343 | 98.4350 |    173M |   31.5G | 2585.240ms  | 53.992ms |   224      |  0.9      |   bicubic    |
| Cswin_base_384         | 85.5100 | 97.4840 | 89.1752 | 98.2813 |    78M  |   47G   | 5725.536ms  | 53.488ms |   384      |  0.9      |   bicubic    |
| Cswin_large_384        | 87.4860 | 98.3460 | 90.2065 | 98.6250 |    173M |   96.8G | 8945.078ms  | 53.283ms |   384      |  0.9      |   bicubic    | 
| Resmlp_12              | 76.6080 | 93.1420 | 83.5323 | 95.8345 |    15M  |   3.0G  | 167.814ms   | 3.507ms  |   224      |  0.9      |   bicubic    |
| Resmlp_12_dist         | 77.9360 | 93.6400 | 84.7258 | 96.2636 |      |     | 167.764ms  | 3.513ms  |   224      |  0.9      |   bicubic    |
| Resmlp_24              | 79.3580 | 94.5320 | 85.2723 | 96.5241 |    30M  |   6.0G  | 321.800ms   | 6.857ms  |   224      |  0.9      |   bicubic    |
| Resmlp_24_dist         | 80.7720 | 95.2180 | 86.5790 | 97.1326 |      |     | 324.436ms   | 6.880ms  |   224      |  0.9      |   bicubic    |
| Resmlp_24_dino         |         |         |         |         |      |     | 322.971ms   | 6.800ms  |   224      |  0.9      |   bicubic    |
| Resmlp_36              | 79.7420 | 94.8860 | 85.6310 | 96.8145 |    45M  |   8.9G  |  484.276ms      | 10.043ms   |   224      |  0.9      |   bicubic    |
| Resmlp_36_dist         | 81.0880 | 95.5820 | 86.9484 | 97.3184 |      |     | 489.108ms   | 10.124ms |   224      |  0.9      |   bicubic    |
| ResmlpB_24             | 80.9440 | 95.0760 | 85.6203 | 96.5006 |      |     | 3159.355ms  | 25.742ms |   224      |  0.9      |   bicubic    |
| ResmlpB_24_in22k       | 84.3960 | 97.1580 | 88.9489 | 98.2855 |      |     | 3155.076ms  | 25.712ms |   224      |  0.9      |   bicubic    |
| ResmlpB_24_dist        | 83.6800 | 96.6740 | 88.4835 | 97.9653 |      |     | 3165.195ms  | 25.983ms |   224      |  0.9      |   bicubic    |



| Metrics  | Oneflow | FlowVision |
|:-:|:-------:|:----------:|
|Acc|0.6.0.dev20211205+cu102|0.0.51|
|Latency|0.6.0dev20211231+cu102|0.0.54|

Note:
- For latency Metrics, CPU and GPU were tested on I9-9900KF with 32G RAM and RTX 2080ti with 11G VRAM respectively.
- The batchsize is 1 for all speed tests.
- For a fair comparison, different models may take in different input size. For example, the input size of Swin_tiny_patch4_window7_224 is 224 while it's 384 for Swin_base_patch4_window12_384.
