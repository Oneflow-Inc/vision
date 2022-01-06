flowvision.models
##############################
Pretrain Models for Visual Tasks


Classification
==============

The models subpackage contains definitions for the following model

Test by lixiang

architectures for image classification:

-  `AlexNet`_
-  `SqueezeNet`_
-  `VGG`_
-  `GoogLeNet`_
-  `InceptionV3`_
-  `ResNet`_
-  `ResNeXt`_
-  `DenseNet`_
-  `ShuffleNetV2`_
-  `MobileNetV2`_
-  `MobileNetV3`_
-  `MNASNet`_
-  `GhostNet`_
-  `Res2Net`_
-  `EfficientNet`_
-  `ReXNet`_
-  `ViT`_
-  `PVT`_
-  `Swin-Transformer`_
-  `CSwin-Transformer`_
-  `CrossFormer`_
-  `Mlp_Mixer`_
-  `ResMLP`_
-  `gMLP`_
-  `ConvMixer`_


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
.. _Res2Net: https://arxiv.org/abs/1904.01169
.. _ReXNet: https://arxiv.org/abs/2007.00992
.. _MNASNet: https://arxiv.org/abs/1807.11626
.. _GhostNet: https://arxiv.org/abs/1911.11907
.. _ViT: https://arxiv.org/abs/2010.11929
.. _PVT: https://arxiv.org/abs/2102.12122
.. _ResMLP: https://arxiv.org/abs/2105.03404
.. _Swin-Transformer: https://arxiv.org/abs/2103.14030
.. _CSwin-Transformer: https://arxiv.org/abs/2107.00652
.. _CrossFormer: https://arxiv.org/abs/2108.00154
.. _Mlp_Mixer: https://arxiv.org/abs/2105.01601
.. _ResMLP: https://arxiv.org/abs/2105.03404
.. _gMLP: https://arxiv.org/abs/2105.08050
.. _ConvMixer: https://openreview.net/pdf?id=TVHS5Y4dNvM
.. _EfficientNet: https://arxiv.org/abs/1905.11946

.. currentmodule:: flowvision.models

Alexnet
-------
.. automodule:: flowvision.models
    :members: alexnet
    

SqueezeNet
----------
.. automodule:: flowvision.models
    :members: 
        squeezenet1_0,
        squeezenet1_1,


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


GoogLeNet
---------
.. automodule:: flowvision.models
    :members: 
        googlenet,


InceptionV3
---------
.. automodule:: flowvision.models
    :members: 
        inception_v3,


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


DenseNet
--------
.. automodule:: flowvision.models
    :members: 
        densenet121,
        densenet169,
        densenet201,
        densenet161,


ShuffleNetV2
------------
.. automodule:: flowvision.models
    :members:
        shufflenet_v2_x0_5,
        shufflenet_v2_x1_0,
        shufflenet_v2_x1_5,
        shufflenet_v2_x2_0,

MobileNetV2
-----------
.. automodule:: flowvision.models
    :members:
        mobilenet_v2


MobileNetV3
-----------
.. automodule:: flowvision.models
    :members:
        mobilenet_v3_small,
        mobilenet_v3_large,


MNASNet
-------
.. automodule:: flowvision.models
    :members:
        mnasnet0_5,
        mnasnet0_75,
        mnasnet1_0,
        mnasnet1_3,


GhostNet
--------
.. automodule:: flowvision.models
    :members:
        ghostnet,


Res2Net
-------
.. automodule:: flowvision.models
    :members:
        res2net50_26w_4s,
        res2net50_26w_6s,
        res2net50_26w_8s,
        res2net50_48w_2s,
        res2net50_14w_8s,
        res2net101_26w_4s,


EfficientNet
------------
.. automodule:: flowvision.models
    :members:
        efficientnet_b0,
        efficientnet_b1,
        efficientnet_b2,
        efficientnet_b3,
        efficientnet_b4,
        efficientnet_b5,
        efficientnet_b6,
        efficientnet_b7


ReXNet
------
.. automodule:: flowvision.models
    :members:
        rexnetv1_1_0,
        rexnetv1_1_3,
        rexnetv1_1_5,
        rexnetv1_2_0,
        rexnetv1_3_0,
        rexnet_lite_1_0,
        rexnet_lite_1_3,
        rexnet_lite_1_5,
        rexnet_lite_2_0,


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


Swin-Transformer
----------------
.. automodule:: flowvision.models
    :members: 
        swin_tiny_patch4_window7_224,
        swin_small_patch4_window7_224,
        swin_base_patch4_window7_224,
        swin_base_patch4_window12_384,
        swin_base_patch4_window7_224_in22k_to_1k,
        swin_base_patch4_window12_384_in22k_to_1k,
        swin_large_patch4_window7_224_in22k_to_1k,
        swin_large_patch4_window12_384_in22k_to_1k,


CSwin-Transformer
-----------------
.. automodule:: flowvision.models
    :members: 
        cswin_tiny_224,
        cswin_small_224,
        cswin_base_224,
        cswin_large_224,
        cswin_base_384,
        cswin_large_384,


CrossFormer
-----------
.. automodule:: flowvision.models
    :members: 
        crossformer_tiny_patch4_group7_224,
        crossformer_small_patch4_group7_224,
        crossformer_base_patch4_group7_224,
        crossformer_large_patch4_group7_224,


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


ConvMixer
---------
.. automodule:: flowvision.models
    :members: 
        convmixer_1536_20,
        convmixer_768_32_relu,
        convmixer_1024_20,



Neural Style
============

.. currentmodule:: flowvision.models
.. automodule:: flowvision.models
    :members:
        neural_style_transfer,


