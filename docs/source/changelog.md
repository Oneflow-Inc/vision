## Changelog

### v0.1.0 (xx/01/2022)

**New Features**

- Support `trunc_normal_` in `flowvision.layers.weight_init` [#92](https://github.com/Oneflow-Inc/vision/pull/92)
- Support [DeiT](https://arxiv.org/abs/2012.12877) model [#115](https://github.com/Oneflow-Inc/vision/pull/115)
- Support `PolyLRScheduler` and `TanhLRScheduler` in `flowvision.scheduler` [#85](https://github.com/Oneflow-Inc/vision/pull/85)
- Add `resmlp_12_224_dino` model and pretrained weight [#128](https://github.com/Oneflow-Inc/vision/pull/128)

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


**Contributors**

A total of 5 developers contributed to this release. Thanks @rentainhe, @simonJJJ, @kaijieshi7, @lixiang007666, @Ldpe2G

