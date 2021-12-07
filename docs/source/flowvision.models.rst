flowvision.models
##############################
Pretrain Models for Visual Tasks


Classification
==============

The models subpackage contains definitions for the following model
architectures for image classification:

-  `AlexNet`_
-  `VGG`_
-  `ResNet`_
-  `SqueezeNet`_
-  `DenseNet`_
-  `InceptionV3`_
-  `GoogLeNet`_
-  `ShuffleNetV2`_
-  `MobileNetV2`_
-  `MobileNetV3`_
-  `ResNeXt`_
-  `MNASNet`_
-  `ViT`_
-  `PVT`_


.. _AlexNet: https://arxiv.org/abs/1404.5997
.. _VGG: https://arxiv.org/abs/1409.1556
.. _ResNet: https://arxiv.org/abs/1512.03385
.. _SqueezeNet: https://arxiv.org/abs/1602.07360
.. _DenseNet: https://arxiv.org/abs/1608.06993
.. _InceptionV3: https://arxiv.org/abs/1512.00567
.. _GoogLeNet: https://arxiv.org/abs/1409.4842
.. _ShuffleNetV2: https://arxiv.org/abs/1807.11164
.. _MobileNetV2: https://arxiv.org/abs/1801.04381
.. _MobileNetV3: https://arxiv.org/abs/1905.02244
.. _ResNeXt: https://arxiv.org/abs/1611.05431
.. _MNASNet: https://arxiv.org/abs/1807.11626
.. _ViT: https://arxiv.org/abs/2010.11929

.. currentmodule:: flowvision.models

Alexnet
-------

.. automodule:: flowvision.models
    :members: alexnet
    

VGG
---

.. automodule:: flowvision.models
    :members: 
        vgg11,
        vgg11_bn,
        vgg13,
        vgg13_bn
        vgg16,
        vgg16_bn,
        vgg19,
        vgg19_bn,


ResNet
------

.. automodule:: flowvision.models
    :members: 
        resnet18,
        resnet34,
        resnet50,
        resnet101,
        resnet152,
        resnext50_32x4d,
        resnext101_32x8d,
        wide_resnet50_2,
        wide_resnet101_2,


ViT
------

.. automodule:: flowvision.models
    :members: 
        vit_b_16_224,
        vit_b_16_384,
        vit_b_32_224,
        vit_b_32_384,
        vit_l_16_384,
        vit_l_32_384,

PVT
------

.. automodule:: flowvision.models
    :members: 
        pvt_tiny,
        pvt_small,
        pvt_medium,
        pvt_large,


ResMLP
------

.. automodule:: flowvision.models
    :members: 
        resmlp_12,
        resmlp_12_dist,
        resmlp_24,
        resmlp_24_dist,


Neural Style
============

.. currentmodule:: flowvision.models
.. automodule:: flowvision.models
    :members:
        neural_style_transfer,

        
