
import sys
import re
import fnmatch
from collections import defaultdict
from copy import deepcopy


pretrained_model_table = defaultdict(set)

def register_model(fn):
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    model_name = fn.__name__

    has_pretrained = False
    if hasattr(mod, 'model_urls') and model_name in mod.model_urls:
        has_pretrained = True if mod.model_urls[model_name] else False

    pretrained_model_table[model_name] = has_pretrained

    return fn

def model_table():
    from rich.table import Table
    from rich import print
    table = Table(title="Models")
    table.add_column("Name", justify="left", no_wrap=True)
    table.add_column("Pretrained", justify="left", no_wrap=True)

    [
        table.add_row(k, "true" if pretrained_model_table[k] else "false") for k in pretrained_model_table.keys()
    ]
    
    print(table)