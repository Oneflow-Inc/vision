import gc
import functools
import sys
import json

def ci_settings(compare=None, retry={"stddev": 1.8, "times": 3}, enable_gc=False):
    def decorator(func):
        func_name = func.__name__
        file_name = sys._getframe().f_back.f_code.co_filename
        print('oneflow-benchmark-function::', end='')
        collect_info = json.dumps({
            'func_name': func_name,
            'file_name': file_name,
            'compare': compare,
            'retry': retry
            })
        print(collect_info)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if(enable_gc):
                gc.collect()
            return func(*args, **kwargs)
        return wrapper
    return decorator