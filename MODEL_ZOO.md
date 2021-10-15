## Model Zoo
Here we provide our test results of pretrained model on ImageNet2012, all tests were done using single TiTanV GPU with batch_size 64

|                        |    top1 |    top5 |  batch_size |
|:-----------------------|--------:|--------:|------------:|
| alexnet                | 56.4798 | 79.0281 |     64      |
| vgg11                  | 69.0018 | 88.6029 |     64      |
| vgg13                  | 69.9029 | 89.2203 |     64      |
| vgg16                  | 71.5533 | 90.3553 |     64      |
| vgg19                  | 72.3485 | 90.8488 |     64      |
| vgg11_bn               | 72.3485 | 90.8488 |     64      |
| vgg13_bn               | 72.3485 | 90.8488 |     64      |
| vgg16_bn               | 72.3485 | 90.8488 |     64      |
| vgg19_bn               | 72.3485 | 90.8488 |     64      |
| densenet121            | 74.4106 | 91.9557 |     64      |
| densenet169            | 75.5635 | 92.7949 |     64      |
| densenet201            | 76.8702 | 93.3584 |     64      |
| densenet169            | 77.1120 | 93.5422 |     64      |