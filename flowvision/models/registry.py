
import sys
import re
import fnmatch
from collections import defaultdict
from copy import deepcopy
import oneflow as flow

"""
1. 判断模型是否存在
2. 判断模型是否有pretrained weight
3. 返回模型
4. 模糊匹配，并且生成对应模糊匹配的模型是否有权重的一个table
"""

class ModelCreator(object):
    _model_list = defaultdict(set)
    _model_entrypoints = {}

    @staticmethod
    def register_model(fn):
        mod = sys.modules[fn.__module__]
        module_name_split = fn.__module__.split('.')
        module_name = module_name_split[-1] if len(module_name_split) else ''

        model_name = fn.__name__

        ModelCreator._model_entrypoints[model_name] = fn

        has_pretrained = False
        if hasattr(mod, 'model_urls') and model_name in mod.model_urls:
            has_pretrained = True if mod.model_urls[model_name] else False

        ModelCreator._model_list[model_name] = has_pretrained

        return fn
    
    @staticmethod
    def create_model(model_name: str, pretrained: bool = False, checkpoint: bool = None):
        if model_name in ModelCreator._model_entrypoints:
            create_fn = ModelCreator._model_entrypoints[model_name]
        else:
            raise RuntimeError('Unknown model (%s)' % model_name)
        model = create_fn(pretrained=pretrained)

        if checkpoint:
            state_dict = flow.load(checkpoint)
            model.load_state_dict(state_dict)
        return model
            

    @staticmethod
    def model_table():
        from rich.table import Table
        from rich import print
        table = Table(title="Models")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Pretrained", justify="left", no_wrap=True)

        [
            table.add_row(k, "true" if ModelCreator._model_list[k] else "false") for k in ModelCreator._model_list.keys()
        ]
        
        print(table)