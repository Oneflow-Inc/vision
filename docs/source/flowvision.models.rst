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
-  `ResMLP`_


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
.. _PVT: https://arxiv.org/abs/2102.12122
.. _ResMLP: https://arxiv.org/abs/2105.03404

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


ShuffleNetV2
------------

.. automodule:: flowvision.models
    :members:
        shufflenet_v2_x0_5,
        shufflenet_v2_x1_0,
        shufflenet_v2_x1_5,
        shufflenet_v2_x2_0,

GhostNet
--------

.. automodule:: flowvision.models
    :members:
        ghostnet,



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


Mlp-Mixer
---------
.. automodule:: flowvision.models
    :members: 
        mlp_mixer_s16_224,
        mlp_mixer_s32_224,
        mlp_mixer_b16_224,
        mlp_mixer_b32_224,
        mlp_mixer_b16_224_in21k,
        mlp_mixer_l16_224,
        mlp_mixer_l32_224,
        mlp_mixer_l16_224_in21k,
        mlp_mixer_b16_224_miil,
        mlp_mixer_b16_224_miil_in21k,


ResMLP
------

.. automodule:: flowvision.models
    :members: 
        resmlp_12,
        resmlp_12_dist,
        resmlp_24,
        resmlp_24_dist,
        resmlp_24_dino,
        resmlp_36,
        resmlp_36_dist,
        resmlpB_24,
        resmlpB_24_in22k,
        resmlpB_24_dist,


gMLP
----
.. automodule:: flowvision.models
    :members: 
        gmlp_ti16_224,
        gmlp_s16_224,
        gmlp_b16_224,



Neural Style
============

.. currentmodule:: flowvision.models
.. automodule:: flowvision.models
    :members:
        neural_style_transfer,

        
