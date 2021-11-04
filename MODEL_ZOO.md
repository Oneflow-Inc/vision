## Model Zoo
Here we provide our test results of pretrained model on ImageNet2012, all tests were done using single TiTanV GPU with batch_size 64 under the same data transformation. There may be some difference with the accuracy of the official pretrained weight because of the test environments.

|                        |    top1 |    top5 |  batch_size | image size |
|:-----------------------|--------:|--------:|------------:|-----------:|
| alexnet                | 56.4798 | 79.0281 |     64      |   224      |
| vgg11                  | 69.0018 | 88.6029 |     64      |   224      |
| vgg13                  | 69.9029 | 89.2203 |     64      |   224      |
| vgg16                  | 71.5533 | 90.3553 |     64      |   224      |
| vgg19                  | 72.3485 | 90.8488 |     64      |   224      |
| vgg11_bn               | 70.3505 | 89.7838 |     64      |   224      |
| vgg13_bn               | 71.5533 | 90.3473 |     64      |   224      |
| vgg16_bn               | 73.3316 | 91.4942 |     64      |   224      |
| vgg19_bn               | 74.1948 | 91.8139 |     64      |   224      |
| resnet18               | 69.7450 | 89.0645 |     64      |   224      |
| resnet34               | 73.2976 | 91.4102 |     64      |   224      |
| resnet50               | 76.0989 | 92.8389 |     64      |   224      |
| resnet101              | 77.3477 | 93.5342 |     64      |   224      |
| resnet152              | 78.2848 | 94.0457 |     64      |   224      |
| resnext-50-32x4d       | 77.5855 | 93.6921 |     64      |   224      |
| resnext-101-32x8d      | 79.2719 | 94.5073 |     64      |   224      |
| wide resnet-50-2       | 78.4467 | 94.0737 |     64      |   224      |
| wide resnet-101-2      | 78.8183 | 94.2775 |     64      |   224      |
| densenet121            | 74.4106 | 91.9557 |     64      |   224      |
| densenet169            | 75.5635 | 92.7949 |     64      |   224      |
| densenet201            | 76.8702 | 93.3584 |     64      |   224      |
| densenet169            | 77.1120 | 93.5422 |     64      |   224      |
| googlenet              | 71.9849 | 90.9047 |     64      |   224      |
| inception_v3           | 77.4337 | 93.5842 |     64      |   299      |
| squeezenet 1.0         | 58.0543 | 80.3848 |     64      |   224      |
| squeezenet 1.1         | 58.1342 | 80.5826 |     64      |   224      |
| shufflenet_v2_x0_5     | 60.5059 | 81.7096 |     64      |   224      |
| shufflenet_v2_x1_0     | 69.3195 | 88.2912 |     64      |   224      |
| mobilenet_v2           | 71.8450 | 90.2653 |     64      |   224      |
| mobilenet_v3_small     | 67.6391 | 87.3781 |     64      |   224      |
| mobilenet_v3_large     | 74.0070 | 91.3243 |     64      |   224      |
| mnasnet 0.5            | 69.6930 | 87.4480 |     64      |   224      |
| mnasnet 1.0            | 73.4215 | 91.4942 |     64      |   224      |
| vit-b-16-384           | 84.1672 | 97.1527 |     64      |   384      |
| vit-b-32-384           | 81.7116 | 96.1217 |     64      |   384      |
| vit-l-16-384           | 85.0444 | 97.3605 |     64      |   384      |
| vit-l-32-384           | 81.5217 | 96.0518 |     64      |   384      |
| convmixer_768_32_relu  | 80.0764 | 94.9896 |     64      |   224      |
| convmixer_1024_20      | 78.4127 | 94.2895 |     64      |   224      |
| convmixer_1536_20      | 81.0461 | 95.6194 |     64      |   224      |
| swin_tiny_patch4_window7_224  | 81.1641 | 95.5003 |  64   |   224     |
| swin_small_patch4_window7_224 | 83.1582 | 96.2376 |  64   |   224     |
| swin_base_patch4_window7_224  | 83.4039 | 96.4434 |  64   |   224     |
| swin_base_patch4_window12_384 | 84.4569 | 96.8950 |  64   |   384     |
| pvt_tiny               | 74.7163 | 92.1595 |     64      |   224      |
| pvt_small              | 79.4697 | 94.7750 |     64      |   224      |
| pvt_medium             | 80.8564 | 95.5103 |     64      |   224      |
| pvt_large              | 81.5477 | 95.6482 |     64      |   224      |
| swin_base_patch4_window7_224_in22k_to_1k   | 85.1223 | 97.4744 |  64  |  224  |
| swin_base_patch4_window12_384_in22k_to_1k  | 86.4330 | 98.0619 |  64  |  384  |
| swin_large_patch4_window7_224_in22k_to_1k  | 86.2492 | 97.8800 |  64  |  224  |
| swin_large_patch4_window12_384_in22k_to_1k | 87.1241 | 98.2326 |  64  |  224  |
| crossformer_tiny_patch4_group7_224  | 81.1461 | 95.3105 |  64  | 224  |
| crossformer_small_patch4_group7_224 | 82.2410 | 95.9579 |  64  | 224  |  
| crossformer_base_patch4_group7_224  | 83.2681 | 96.4694 |  64  | 224  |
| crossformer_large_patch4_group7_224 | 83.7516 | 96.4954 |  64  | 224  |
| cswin_tiny_224         | 80.5147 | 95.1746 |    64       |   224      |
| cswin_small_224        | 81.7355 | 95.6422 |    64       |   224      |
| cswin_base_224         | 83.5458 | 96.5733 |    64       |   224      |
| cswin_large_224        | 85.7697 | 97.7362 |    64       |   224      |
| cswin_base_384         | - | - |    64       |   224      |
| cswin_large_384        | - | - |    64       |   224      |
