# vision
Datasets, Transforms and Models specific to Computer Vision


## Installation
- First install the nightly version of `OneFlow`
```bash
python3 -m pip install oneflow -f https://staging.oneflow.info/branch/master/cu102
```

- Then install the latest stable release of `flowvision`
```bash
pip install flowvision==0.0.5
```

- Or install the nightly release of `flowvision`
```bash
pip install -i https://test.pypi.org/simple/ flowvision==0.0.5
```

## Supported Model
All of the supported models can be found in our model summary page [here](MODEL_SUMMARY.md).


## Usage
<details>
<summary> <b> Quick Start </b> </summary>

- list supported model
```python
from flowvision import ModelCreator
ModelCreator.model_table()
```

- search supported model by wildcard
```python
from flowvision import ModelCreator
ModelCreator.model_table("*vit*", pretrained=True)
ModelCreator.model_table("*vit*", pretrained=False)
ModelCreator.model_table('alexnet')
```

- create model use `ModelCreator`
```python
from flowvision import ModelCreator
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
ModelCreator.model_table()
```
```
           Models            
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name         ┃ Pretrained ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ alexnet      │ true       │
│ vit_b_16_224 │ false      │
│ vit_b_16_384 │ true       │
│ vit_b_32_224 │ false      │
│ vit_b_32_384 │ true       │
│ vit_l_16_384 │ true       │
│ vit_l_32_384 │ true       │
└──────────────┴────────────┘
```
show all of the supported model in the table manner

- List models with pretrained weights
```python
from flowvision.models import ModelCreator
ModelCreator.model_table(pretrained=True)
```
```
           Models            
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name         ┃ Pretrained ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ alexnet      │ true       │
│ vit_b_16_384 │ true       │
│ vit_b_32_384 │ true       │
│ vit_l_16_384 │ true       │
│ vit_l_32_384 │ true       │
└──────────────┴────────────┘
```
- Search for model by Wildcard
```python
from flowvision.models import ModelCreator
ModelCreator.model_table('vit*')
```
```
           Models            
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name         ┃ Pretrained ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ vit_b_16_224 │ false      │
│ vit_b_16_384 │ true       │
│ vit_b_32_224 │ false      │
│ vit_b_32_384 │ true       │
│ vit_l_16_384 │ true       │
│ vit_l_32_384 │ true       │
└──────────────┴────────────┘
```
- Search for model with pretrained weights by Wildcard
```python
from flowvision.models import ModelCreator
ModelCreator.model_table('vit*', pretrained=True)
```
```
           Models            
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name         ┃ Pretrained ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ vit_b_16_384 │ true       │
│ vit_b_32_384 │ true       │
│ vit_l_16_384 │ true       │
│ vit_l_32_384 │ true       │
└──────────────┴────────────┘
```

</details>

## Model Zoo
We have conducted all the tests under the same setting, please refer to the model page [here](MODEL_ZOO.md) for more details.

## Disclaimer on Datasets
This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!
