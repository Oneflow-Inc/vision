# flowvision
The flowvision package consists of popular datasets, SOTA computer vision models, layers, utilities, schedulers, advanced data augmentations and common image transformations based on OneFlow.


## Installation
First install OneFlow, please refer to [install-oneflow](https://github.com/Oneflow-Inc/oneflow#install-oneflow) for more details.

Then install the latest stable release of `flowvision`
```bash
pip install flowvision==0.0.56
```

## Overview of flowvision structure
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Vision Models</b>
      </td>
      <td>
        <b>Layers</b>
      </td>
      <td>
        <b>Components</b>
      </td>
      <td>
        <b>Data Augmentation</b>
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
      </ul>
      </td>
      <td>
      <ul><li><b>Attention Layer</b></li>
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
      <ul><li><b>Regularization Layer</b></li>
          <ul>
            <li>Drop Block</li>
            <li>Drop Path</li>
            <li>Stochastic Depth</li>
          </ul>  
        </ul>
      <ul><li><b>Basic Layer</b></li>
          <ul>
            <li>Patch Embedding</li>
            <li>Mlp Block</li>
            <li>FPN</li>
          </ul>  
        </ul>
      <ul><li><b>Activation Layer</b></li>
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
      </td>
      <td>
        <ul><li><b>Common</b></li>
          <ul>
            <li>Sync-BN</li>
            <li>Group Norm</li>
            <li>DCNv2</li>
            <li>Non-local</li>
          </ul>  
        </ul>
        <ul><li><b>KeyPoint</b></li>
          <ul>
            <li>DarkPose</li>
          </ul>  
        </ul>
        <ul><li><b>FPN</b></li>
          <ul>
            <li>BiFPN</li>
            <li>BFP</li>  
            <li>HRFPN</li>
            <li>ACFPN</li>
          </ul>  
        </ul>  
        <ul><li><b>Loss</b></li>
          <ul>
            <li>Smooth-L1</li>
            <li>GIoU/DIoU/CIoU</li>  
            <li>IoUAware</li>
          </ul>  
        </ul>  
        <ul><li><b>Post-processing</b></li>
          <ul>
            <li>SoftNMS</li>
            <li>MatrixNMS</li>  
          </ul>  
        </ul>
        <ul><li><b>Speed</b></li>
          <ul>
            <li>FP16 training</li>
            <li>Multi-machine training </li>  
          </ul>  
        </ul>  
      </td>
      <td>
        <ul>
          <li>Resize</li>  
          <li>Lighting</li>  
          <li>Flipping</li>  
          <li>Expand</li>
          <li>Crop</li>
          <li>Color Distort</li>  
          <li>Random Erasing</li>  
          <li>Mixup </li>
          <li>Mosaic</li>
          <li>Cutmix </li>
          <li>Grid Mask</li>
          <li>Auto Augment</li>  
          <li>Random Perspective</li>  
        </ul>  
      </td>  
    </tr>


</td>
    </tr>
  </tbody>
</table>


## Documentation
You can find the API documentation on the website: https://flowvision.readthedocs.io/en/latest/index.html

## Model Zoo
All of the supported models can be found in our model summary page [here](MODEL_SUMMARY.md).

We have conducted all the tests under the same setting, please refer to the model page [here](MODEL_ZOO.md) for more details.

## Quick Start
<details>
<summary> <b> Quick Start </b> </summary>

- list supported model
```python
from flowvision.models import ModelCreator
supported_model_table = ModelCreator.model_table()
print(supported_model_table)
```

- search supported model by wildcard
```python
from flowvision.models import ModelCreator
pretrained_vit_model = ModelCreator.model_table("*vit*", pretrained=True)
supported_vit_model = ModelCreator.model_table("*vit*", pretrained=False)
supported_alexnet_model = ModelCreator.model_table('alexnet')

# check the model table
print(pretrained_vit_model)
print(supported_vit_model)
print(supported_alexnet_model)
```

- create model use `ModelCreator`
```python
from flowvision.models import ModelCreator
model = ModelCreator.create_model('alexnet', pretrained=True)
```

</details>

<details>
<summary> <b> ModelCreator </b> </summary>

- Create model in a simple way
```python
from flowvision.models import ModelCreator
model = ModelCreator.create_model('alexnet', pretrained=True)
```
the pretrained weight will be saved to `./checkpoints`

- Supported model table
```python
from flowvision.models import ModelCreator
supported_model_table = ModelCreator.model_table()
print(supported_model_table)
```
```
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
│ shufflenet_v2_x0_5                         │ true         │
├────────────────────────────────────────────┼──────────────┤
│ shufflenet_v2_x1_0                         │ true         │
├────────────────────────────────────────────┼──────────────┤
│ shufflenet_v2_x1_5                         │ false        │
├────────────────────────────────────────────┼──────────────┤
│ shufflenet_v2_x2_0                         │ false        │
├────────────────────────────────────────────┼──────────────┤
│                    ...                     │     ...      │
├────────────────────────────────────────────┼──────────────┤
│ wide_resnet101_2                           │ true         │
├────────────────────────────────────────────┼──────────────┤
│ wide_resnet50_2                            │ true         │
╘════════════════════════════════════════════╧══════════════╛
```
show all of the supported model in the table manner

- Check the table of the models with pretrained weights.
```python
from flowvision.models import ModelCreator
pretrained_model_table = ModelCreator.model_table(pretrained=True)
print(pretrained_model_table)
```
```
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
│                    ...                     │     ...      │
├────────────────────────────────────────────┼──────────────┤
│ wide_resnet101_2                           │ true         │
├────────────────────────────────────────────┼──────────────┤
│ wide_resnet50_2                            │ true         │
╘════════════════════════════════════════════╧══════════════╛
```
- Search for model by Wildcard.
```python
from flowvision.models import ModelCreator
supported_vit_model = ModelCreator.model_table('vit*')
print(supported_vit_model)
```
```
╒════════════════════╤══════════════╕
│ Supported Models   │ Pretrained   │
╞════════════════════╪══════════════╡
│ vit_b_16_224       │ false        │
├────────────────────┼──────────────┤
│ vit_b_16_384       │ true         │
├────────────────────┼──────────────┤
│ vit_b_32_224       │ false        │
├────────────────────┼──────────────┤
│ vit_b_32_384       │ true         │
├────────────────────┼──────────────┤
│ vit_l_16_384       │ true         │
├────────────────────┼──────────────┤
│ vit_l_32_384       │ true         │
╘════════════════════╧══════════════╛
```
- Search for model with pretrained weights by Wildcard.
```python
from flowvision.models import ModelCreator
ModelCreator.model_table('vit*', pretrained=True)
```
```
╒════════════════════╤══════════════╕
│ Supported Models   │ Pretrained   │
╞════════════════════╪══════════════╡
│ vit_b_16_384       │ true         │
├────────────────────┼──────────────┤
│ vit_b_32_384       │ true         │
├────────────────────┼──────────────┤
│ vit_l_16_384       │ true         │
├────────────────────┼──────────────┤
│ vit_l_32_384       │ true         │
╘════════════════════╧══════════════╛
```

</details>

## Disclaimer on Datasets
This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!
