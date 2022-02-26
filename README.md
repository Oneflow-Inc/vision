<h2 align="center">flowvision</h2>
<p align="center">
    <a href="https://pypi.org/project/flowvision/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/flowvision">
    </a>
    <a href="https://flowvision.readthedocs.io/en/latest/index.html">
        <img alt="docs" src="https://img.shields.io/badge/docs-latest-blue">
    </a>
    <a href="https://github.com/Oneflow-Inc/vision/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/Oneflow-Inc/vision.svg?color=blue">
    </a>
    <a href="https://github.com/Oneflow-Inc/vision/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/Oneflow-Inc/vision.svg">
    </a>
    <a href="https://github.com/Oneflow-Inc/vision/issues">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-pink.svg">
    </a>
</p>


## Introduction
The flowvision package consists of popular datasets, SOTA computer vision models, layers, utilities, schedulers, advanced data augmentations and common image transformations based on OneFlow.

## Installation
First install OneFlow, please refer to [install-oneflow](https://github.com/Oneflow-Inc/oneflow#install-oneflow) for more details.

Then install the latest stable release of `flowvision`
```bash
pip install flowvision==0.1.0
```

## Overview of flowvision structure
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Vision Models</b>
      </td>
      <td>
        <b>Components</b>
      </td>
      <td>
        <b>Augmentation and Datasets</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li><b>Classification</b></li>
          <ul>
            <li>AlexNet</li>
            <li>SqueezeNet</li>
            <li>VGG</li>
            <li>GoogleNet</li>
            <li>InceptionV3</li>
            <li>ResNet</li>
            <li>ResNeXt</li>
            <li>ResNeSt</li>
            <li>SENet</li>
            <li>DenseNet</li>
            <li>ShuffleNetV2</li>  
            <li>MobileNetV2</li>
            <li>MobileNetV3</li>
            <li>MNASNet</li>
            <li>Res2Net</li>
            <li>EfficientNet</li>  
            <li>GhostNet</li>
            <li>ReXNet</li>
            <li>Vision Transformer</li>
            <li>DeiT</li>
            <li>PVT</li>
            <li>Swin Transformer</li>
            <li>CSwin Transformer</li>
            <li>CrossFormer</li>
            <li>Mlp Mixer</li>
            <li>ResMLP</li>
            <li>gMLP</li>
            <li>ConvMixer</li>
            <li>ConvNeXt</li>
            <li>RegionViT</li>
            <li>UniFormer</li>
        </ul>
        <li><b>Detection</b></li>
        <ul>
            <li>SSD</li>
            <li>SSDLite</li>
            <li>Faster RCNN</li>
            <li>RetinaNet</li>
        </ul>
        <li><b>Segmentation</b></li>
        <ul>
            <li>FCN</li>
            <li>DeepLabV3</li>
        </ul>
        <li><b>Neural Style Transfer</b></li>
        <ul>
            <li>StyleNet</li>
        </ul>
        <li><b>Face Recognition</b></li>
        <ul>
            <li>IResnet</li>
        </ul>        
      </ul>
      </td>
      <td>
      <ul><li><b>Attention Layers</b></li>
          <ul>
            <li>SE</li>
            <li>BAM</li>
            <li>CBAM</li>
            <li>ECA</li>
            <li>Non Local Attention</li>
            <li>Global Context</li>
            <li>Gated Channel Transform</li>
            <li>Coordinate Attention</li>
          </ul>  
        </ul>
      <ul><li><b>Regularization Layers</b></li>
          <ul>
            <li>Drop Block</li>
            <li>Drop Path</li>
            <li>Stochastic Depth</li>
            <li>LayerNorm2D</li>
          </ul>  
        </ul>
      <ul><li><b>Basic Layers</b></li>
          <ul>
            <li>Patch Embedding</li>
            <li>Mlp Block</li>
            <li>FPN</li>
          </ul>  
        </ul>
      <ul><li><b>Activation Layers</b></li>
          <ul>
            <li>Hard Sigmoid</li>
            <li>Hard Swish</li>
          </ul>  
        </ul>
      <ul><li><b>Initialization Function</b></li>
          <ul>
            <li>Truncated Normal</li>
            <li>Lecun Normal</li>
          </ul>  
        </ul>
      <ul><li><b>LR Scheduler</b></li>
        <ul>
            <li>StepLRScheduler</li>
            <li>MultiStepLRScheduler</li>
            <li>CosineLRScheduler</li>
            <li>LinearLRScheduler</li>
            <li>PolyLRScheduler</li>
            <li>TanhLRScheduler</li>
          </ul>  
        </ul>
        <ul><li><b>Loss</b></li>
          <ul>
            <li>LabelSmoothingCrossEntropy</li>
            <li>SoftTargetCrossEntropy</li>
          </ul>  
        </ul>
      </td>
      <td>
        <ul><li><b>Basic Augmentation</b></li>
          <ul>
            <li>CenterCrop</li>
            <li>RandomCrop</li>
            <li>RandomResizedCrop</li>
            <li>FiveCrop</li>
            <li>TenCrop</li>
            <li>RandomVerticalFlip</li>
            <li>RandomHorizontalFlip</li>
            <li>Resize</li>
          </ul>  
        </ul>
        <ul><li><b>Advanced Augmentation</b></li>
          <ul>
            <li>Mixup</li>
            <li>CutMix</li>
            <li>AugMix</li>
            <li>RandomErasing</li>
            <li>Rand Augmentation</li>
            <li>Auto Augmentation</li>
          </ul>  
        </ul>
        <ul><li><b>Datasets</b></li>
          <ul>
            <li>CIFAR10</li>
            <li>CIFAR100</li>
            <li>COCO</li>
            <li>FashionMNIST</li>
            <li>ImageNet</li>
            <li>VOC</li>
          </ul>  
        </ul>
      </td>  
    </tr>


</td>
    </tr>
  </tbody>
</table>


## Documentation
Please refer to [docs](https://flowvision.readthedocs.io/en/latest/index.html) for full API documentation and tutorials


## ChangeLog
Please refer to [ChangeLog](https://flowvision.readthedocs.io/en/latest/changelog.html) for details and release history


## Model Zoo
We have conducted all the tests under the same setting, please refer to the model page [here](./results/results_imagenet.md) for more details.

## Quick Start
### Create a model
In flowvision we support two ways to create a model.

- First, import the target model from `flowvision.models`, e.g., create `alexnet` from flowvision

```python
from flowvision.models.alexnet import alexnet
model = alexnet()
```

- Second, create model in an easier way by using `ModelCreator`, e.g., create `alexnet` model by `ModelCreator`
```python
from flowvision.models import ModelCreator
alexnet = ModelCreator.create_model("alexnet")
```

- To create a pretrained model, simply pass `pretrained=True` into `ModelCreator.create_model` function
```python
from flowvision.models import ModelCreator
alexnet = ModelCreator.create_model("alexnet", pretrained=True)
```

- To create a custom model to fit different number of classes, simply pass `num_classes=<number of class>` into `ModelCreator.create_model` function
```python
from flowvision.models import ModelCreator
model = ModelCreator.create_model("alexnet", num_classes=100)
```

### Tabulate all models with pretrained weights
`ModelCreator.model_table()` returns a tabular results of available models in `flowvision`. To check all of pretrained models, pass in `pretrained=True` in `ModelCreator.model_table()`.
```python
from flowvision.models import ModelCreator
all_pretrained_models = ModelCreator.model_table(pretrained=True)
print(all_pretrained_models)
```
You can get the results like:
```python
╒════════════════════════════════════════════╤══════════════╕
│ Supported Models                           │ Pretrained   │
╞════════════════════════════════════════════╪══════════════╡
│ alexnet                                    │ true         │
├────────────────────────────────────────────┼──────────────┤
│ convmixer_1024_20                          │ true         │
├────────────────────────────────────────────┼──────────────┤
│ convmixer_1536_20                          │ true         │
├────────────────────────────────────────────┼──────────────┤
│ convmixer_768_32_relu                      │ true         │
├────────────────────────────────────────────┼──────────────┤
│ crossformer_base_patch4_group7_224         │ true         │
├────────────────────────────────────────────┼──────────────┤
│ crossformer_large_patch4_group7_224        │ true         │
├────────────────────────────────────────────┼──────────────┤
│ crossformer_small_patch4_group7_224        │ true         │
├────────────────────────────────────────────┼──────────────┤
│ crossformer_tiny_patch4_group7_224         │ true         │
├────────────────────────────────────────────┼──────────────┤
│                    ...                     │ ...          │
├────────────────────────────────────────────┼──────────────┤
│ wide_resnet101_2                           │ true         │
├────────────────────────────────────────────┼──────────────┤
│ wide_resnet50_2                            │ true         │
╘════════════════════════════════════════════╧══════════════╛
```

### Search for supported model by Wildcard
It is easy to search for model architectures by using Wildcard as below:
```python
from flowvision.models import ModelCreator
all_efficientnet_models = ModelCreator.model_table("**efficientnet**")
print(all_efficientnet_models)
```
You can get the results like:
```python
╒════════════════════╤══════════════╕
│ Supported Models   │ Pretrained   │
╞════════════════════╪══════════════╡
│ efficientnet_b0    │ true         │
├────────────────────┼──────────────┤
│ efficientnet_b1    │ true         │
├────────────────────┼──────────────┤
│ efficientnet_b2    │ true         │
├────────────────────┼──────────────┤
│ efficientnet_b3    │ true         │
├────────────────────┼──────────────┤
│ efficientnet_b4    │ true         │
├────────────────────┼──────────────┤
│ efficientnet_b5    │ true         │
├────────────────────┼──────────────┤
│ efficientnet_b6    │ true         │
├────────────────────┼──────────────┤
│ efficientnet_b7    │ true         │
╘════════════════════╧══════════════╛
```

### List all models supported in flowvision
`ModelCreator.model_list` has similar function as `ModelCreator.model_table` but return a list object, which gives the user a more flexible way to check the supported model in flowvision.
- List all models with pretrained weights
```python
from flowvision.models import ModelCreator
all_pretrained_models = ModelCreator.model_list(pretrained=True)
print(all_pretrained_models[:5])
```
You can get the results like:
```python
['alexnet', 
 'convmixer_1024_20', 
 'convmixer_1536_20', 
 'convmixer_768_32_relu', 
 'crossformer_base_patch4_group7_224']
```

- Support wildcard search
```python
from flowvision.models import ModelCreator
all_efficientnet_models = ModelCreator.model_list("**efficientnet**")
print(all_efficientnet_models)
```
You can get the results like:
```python
['efficientnet_b0', 
 'efficientnet_b1', 
 'efficientnet_b2', 
 'efficientnet_b3', 
 'efficientnet_b4', 
 'efficientnet_b5', 
 'efficientnet_b6', 
 'efficientnet_b7']
```

</details>

## Disclaimer on Datasets
This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!
