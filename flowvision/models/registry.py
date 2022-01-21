"""
Modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/registry.py
"""
import sys
import re
import fnmatch
from collections import defaultdict
from copy import deepcopy
from tabulate import tabulate

import oneflow as flow


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


class ModelCreator(object):
    _model_list = defaultdict(
        set
    )  # only contain model, and if it has pretrained or not, e.g. {'alexnet': True}
    # _model_with_module = defaultdict(set)  # contain model and its module
    _model_entrypoints = {}
    _model_to_module = {}

    @staticmethod
    def register_model(fn):
        mod = sys.modules[fn.__module__]
        module_name_split = fn.__module__.split(".")
        module_name = module_name_split[-1] if len(module_name_split) else ""

        model_name = fn.__name__

        ModelCreator._model_entrypoints[model_name] = fn
        ModelCreator._model_to_module[model_name] = module_name

        has_pretrained = False
        if hasattr(mod, "model_urls") and model_name in mod.model_urls:
            has_pretrained = True if mod.model_urls[model_name] else False

        ModelCreator._model_list[model_name] = has_pretrained

        return fn

    @staticmethod
    def create_model(
        model_name: str, pretrained: bool = False, checkpoint: str = None, **kwargs
    ):
        if model_name in ModelCreator._model_entrypoints:
            create_fn = ModelCreator._model_entrypoints[model_name]
        else:
            raise RuntimeError("Unknown model (%s)" % model_name)
        model = create_fn(pretrained=pretrained, **kwargs)

        if checkpoint is not None:
            state_dict = flow.load(checkpoint)
            model.load_state_dict(state_dict)
        return model

    @staticmethod
    def model_table(filter="", pretrained=False, **kwargs):
        all_models = ModelCreator._model_entrypoints.keys()
        if filter:
            models = []
            include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
            for f in include_filters:
                include_models = fnmatch.filter(all_models, f)
                if len(include_models):
                    models = set(models).union(include_models)
        else:
            models = all_models

        show_dict = {}
        sorted_model = list(sorted(models))
        if pretrained:
            for model in sorted_model:
                if ModelCreator._model_list[model]:
                    show_dict[model] = ModelCreator._model_list[model]
        else:
            for model in sorted_model:
                show_dict[model] = ModelCreator._model_list[model]

        table_headers = ["Supported Models", "Pretrained"]
        table_items = [
            (k, "true" if show_dict[k] else "false") for k in show_dict.keys()
        ]
        table = tabulate(
            table_items, headers=table_headers, tablefmt="fancy_grid", **kwargs
        )
        return table

    @staticmethod
    def model_list(filter="", pretrained=False, **kwargs):
        all_models = ModelCreator._model_entrypoints.keys()
        if filter:
            models = []
            include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
            for f in include_filters:
                include_models = fnmatch.filter(all_models, f)
                if len(include_models):
                    models = set(models).union(include_models)
        else:
            models = all_models

        sorted_model = list(sorted(models))
        if pretrained:
            for model in sorted_model:
                if not ModelCreator._model_list[model]:
                    sorted_model.remove(model)

        return sorted_model

    def __repr__(self) -> str:
        all_model_table = ModelCreator.model_table("")
        return "Registry of all models:\n" + all_model_table

    __str__ = __repr__
