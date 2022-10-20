## Changelog
- [Changelog](#changelog)
  - [V0.3.0](#v030)
  - [V0.2.0 (22/07/2022)](#v020-22072022)
  - [v0.1.0 (10/02/2022)](#v010-10022022)


### V0.3.0

**New Features**
- Support ``transforms.Grayscale`` and ``transforms.Solarization`` transforms [#220](https://github.com/Oneflow-Inc/vision/pull/220)
- Support TorchHub functionality [#258](https://github.com/Oneflow-Inc/vision/pull/258)
- Support ``transforms.RandomAffine`` method [#259](https://github.com/Oneflow-Inc/vision/pull/259)
- Support ``transforms.RandomRotation`` method [#261](https://github.com/Oneflow-Inc/vision/pull/261)

**Bug Fixes**
- fix hsv2rgb bug [#256](https://github.com/Oneflow-Inc/vision/pull/256)
- fix rgb2hsv bug [#257](https://github.com/Oneflow-Inc/vision/pull/257)

**Improvements**

**Docs Update**

**Contributors**
A total of x developers contributed to this release. Thanks @BBuf, @rentainhe, @Flowingsun007


### V0.2.0 

**New Features**
- Support [SENet](https://arxiv.org/abs/1709.01507) model and pretrained weight [#149](https://github.com/Oneflow-Inc/vision/pull/149)
- Support [ResNeSt](https://arxiv.org/abs/2004.08955) model and pretrained weight [#156](https://github.com/Oneflow-Inc/vision/pull/156)
- Support [PoolFormer](https://arxiv.org/abs/2111.11418) model and pretrained weight [#137](https://github.com/Oneflow-Inc/vision/pull/137)
- Support [RegionViT](https://arxiv.org/abs/2106.02689) model and pretrained weight [#144](https://github.com/Oneflow-Inc/vision/pull/144)
- Support [UniFormer](https://arxiv.org/abs/2201.04676) model and pretrained weight [#147](https://github.com/Oneflow-Inc/vision/pull/147)
- Support ``IResNet`` model for face recognition [#160](https://github.com/Oneflow-Inc/vision/pull/160)
- Support [VAN](https://arxiv.org/abs/2202.09741) model and pretrained weight [#166](https://github.com/Oneflow-Inc/vision/pull/166)
- Support [Dynamic convolution](https://arxiv.org/abs/1912.03458) module [#166](https://github.com/Oneflow-Inc/vision/pull/169)
- Support ``transforms.RandomGrayscale`` method [#171](https://github.com/Oneflow-Inc/vision/pull/171)
- Support [RegNet](https://arxiv.org/abs/2003.13678) model and pretrained weight [#166](https://github.com/Oneflow-Inc/vision/pull/166)
- Support [LeViT](https://arxiv.org/abs/2104.01136) model and pretrained weight [#177](https://github.com/Oneflow-Inc/vision/pull/177)
- Support ``transforms.GaussianBlur`` method [#188](https://github.com/Oneflow-Inc/vision/pull/188)
- Support ```SUN397```,```Country211```, dataset [#215](https://github.com/Oneflow-Inc/vision/pull/215)
- Support ```Flowers102```,```FGVCAircraft```,```OxfordIIITPet```,```DTD```,```Food101```,```RenderedSST2```,```StanfordCars```,```PCAM```,```EuroSAT```,```GTSRB```,```CLEVR```,```FER2013``` dataset [#217](https://github.com/Oneflow-Inc/vision/pull/217)
- Support [MobileViT](https://arxiv.org/abs/2110.02178) model and pretrained weight [#231](https://github.com/Oneflow-Inc/vision/pull/231)
- Support [DeiT III](https://arxiv.org/abs/2204.07118) model and pretrained weight [#239](https://github.com/Oneflow-Inc/vision/pull/239)
- Support [CaiT](https://arxiv.org/abs/2103.17239) model and pretrained weight [#239](https://github.com/Oneflow-Inc/vision/pull/239)
- Support [DLA](https://arxiv.org/abs/1707.06484) model and pretrained weight [#239](https://github.com/Oneflow-Inc/vision/pull/239)
- Support [GENet](https://arxiv.org/abs/2006.14090) model and pretrained weight [#250](https://github.com/Oneflow-Inc/vision/pull/250)
- Support [HRNet](https://arxiv.org/abs/1908.07919) model and pretrained weight [#250](https://github.com/Oneflow-Inc/vision/pull/250)
- Support [FAN](https://arxiv.org/abs/2204.12451) model and pretrained weight [#250](https://github.com/Oneflow-Inc/vision/pull/250)

**Bug Fixes**
- Fix benchmark normalize mode error [#146](https://github.com/Oneflow-Inc/vision/pull/146)

**Improvements**

**Docs Update**
- Add `SEModule` Docs [#143](https://github.com/Oneflow-Inc/vision/pull/143)
- Add `Scheduler` and `Data` Docs [#189](https://github.com/Oneflow-Inc/vision/pull/189)

**Contributors**
A total of x developers contributed to this release.

### v0.1.0 (10/02/2022)

**New Features**

- Support `trunc_normal_` in `flowvision.layers.weight_init` [#92](https://github.com/Oneflow-Inc/vision/pull/92)
- Support [DeiT](https://arxiv.org/abs/2012.12877) model [#115](https://github.com/Oneflow-Inc/vision/pull/115)
- Support `PolyLRScheduler` and `TanhLRScheduler` in `flowvision.scheduler` [#85](https://github.com/Oneflow-Inc/vision/pull/85)
- Add `resmlp_12_224_dino` model and pretrained weight [#128](https://github.com/Oneflow-Inc/vision/pull/128)
- Support [ConvNeXt](https://arxiv.org/abs/2201.03545) model [#93](https://github.com/Oneflow-Inc/vision/pull/93)
- Add [ReXNet](https://arxiv.org/abs/2007.00992) weights [#132](https://github.com/Oneflow-Inc/vision/pull/132)


**Bug Fixes**

- Fix `F.normalize` usage in SSD [#116](https://github.com/Oneflow-Inc/vision/pull/116)
- Fix bug in `EfficientNet` and `Res2Net` [#122](https://github.com/Oneflow-Inc/vision/pull/122)
- Fix error pretrained weight usage in `vit_small_patch32_384` and `res2net50_48w_2s` [#128](https://github.com/Oneflow-Inc/vision/pull/128)


**Improvements**

- Refator `trunc_normal_` and `linspace` usage in Swin-T, Cross-Former, PVT and CSWin models [#100](https://github.com/Oneflow-Inc/vision/pull/100)
- Refator `Vision Transformer` model [#115](https://github.com/Oneflow-Inc/vision/pull/115)
- Refine `flowvision.models.ModelCreator` to support `ModelCreator.model_list` func [#123](https://github.com/Oneflow-Inc/vision/pull/123)
- Refator README [#124](https://github.com/Oneflow-Inc/vision/pull/124)
- Refine `load_state_dict_from_url` in `flowvision.models.utils` to support downloading pretrained weights to cache dir `~/.oneflow/flowvision_cache` [#127](https://github.com/Oneflow-Inc/vision/pull/127)
- Rebuild a cleaner model zoo and test all the model with pretrained weights released in flowvision [#128](https://github.com/Oneflow-Inc/vision/pull/128)

**Docs Update**
- Update `Vision Transformer` docs [#115](https://github.com/Oneflow-Inc/vision/pull/115)
- Add `Getting Started` docs [#124](https://github.com/Oneflow-Inc/vision/pull/124)
- Add `resmlp_12_224_dino` docs [#128](https://github.com/Oneflow-Inc/vision/pull/128)
- Fix `VGG` docs bug [#128](https://github.com/Oneflow-Inc/vision/pull/128)
- Add `ConvNeXt` docs [#93](https://github.com/Oneflow-Inc/vision/pull/93)


**Contributors**

A total of 5 developers contributed to this release. Thanks @rentainhe, @simonJJJ, @kaijieshi7, @lixiang007666, @Ldpe2G

