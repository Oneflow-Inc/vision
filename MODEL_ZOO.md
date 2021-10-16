## Model Zoo
Here we provide our test results of pretrained model on ImageNet2012, all tests were done using single TiTanV GPU with batch_size 64

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
| densenet121            | 74.4106 | 91.9557 |     64      |   224      |
| densenet169            | 75.5635 | 92.7949 |     64      |   224      |
| densenet201            | 76.8702 | 93.3584 |     64      |   224      |
| densenet169            | 77.1120 | 93.5422 |     64      |   224      |
| mnasnet 0.5            | 69.6930 | 87.4480 |     64      |   224      |
| mnasnet 1.0            | 73.4215 | 91.4942 |     64      |   224      |