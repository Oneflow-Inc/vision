## Changelog

### v0.1.0 (xx/01/2022)

**New Features**

- Support `trunc_normal_` in `flowvision.layers.weight_init` [#92](https://github.com/Oneflow-Inc/vision/pull/92)
- Support [DeiT](https://arxiv.org/abs/2012.12877) model [#115](https://github.com/Oneflow-Inc/vision/pull/115)
- Support `PolyLRScheduler` and `TanhLRScheduler` in `flowvision.scheduler` [#85](https://github.com/Oneflow-Inc/vision/pull/85)

**Bug Fixes**

- Fix `F.normalize` usage in SSD [#116](https://github.com/Oneflow-Inc/vision/pull/116)
- Fix bug in `EfficientNet` and `Res2Net` [#122](https://github.com/Oneflow-Inc/vision/pull/122)

**Improvements**

- Refator `trunc_normal_` and `linspace` usage in Swin-T, Cross-Former, PVT and CSWin models [#100](https://github.com/Oneflow-Inc/vision/pull/100)
- Refator `Vision Transformer` model [#115](https://github.com/Oneflow-Inc/vision/pull/115)
- Refine `flowvision.models.ModelCreator` to support `ModelCreator.model_list` func [#123](https://github.com/Oneflow-Inc/vision/pull/123)


**Docs Update**
- Update `Vision Transformer` docs [#115](https://github.com/Oneflow-Inc/vision/pull/115)


**Contributors**

A total of 5 developers contributed to this release. Thanks @rentainhe, @simonJJJ, @kaijieshi7, @lixiang007666, @Ldpe2G

