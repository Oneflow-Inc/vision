import sys
import re
import fnmatch
from collections import defaultdict
from copy import deepcopy
import oneflow as flow


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
        model_name: str, pretrained: bool = False, checkpoint: bool = None
    ):
        if model_name in ModelCreator._model_entrypoints:
            create_fn = ModelCreator._model_entrypoints[model_name]
        else:
            raise RuntimeError("Unknown model (%s)" % model_name)
        model = create_fn(pretrained=pretrained)

        if checkpoint:
            state_dict = flow.load(checkpoint)
            model.load_state_dict(state_dict)
        return model

    @staticmethod
    def model_table(filter="", pretrained=False):
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

        # ModelCreator._model_list

        from rich.table import Table
        from rich import print

        table = Table(title="Models")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Pretrained", justify="left", no_wrap=True)

        [
            table.add_row(k, "true" if show_dict[k] else "false")
            for k in show_dict.keys()
        ]

        print(table)
