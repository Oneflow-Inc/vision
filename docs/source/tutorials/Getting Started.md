# Getting Started

## Installation
- To install latest stable release of flowvision:
```bash
pip install flowvision==0.0.56
```
- For an editable install
```bash
git clone https://github.com/Oneflow-Inc/vision.git
cd vision
pip install -e .
```

## Usage
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
