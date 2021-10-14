# vision
Datasets, Transforms and Models specific to Computer Vision



## Usage
### ModelCreator
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
