from .layer_registry import Layer_Registry

LAYER_REGISTRY = Layer_Registry("flowvision layers")
LAYER_REGISTRY.__doc__ = """
Register all the layers in flowvision into LAYER_REGISTRY for easy usage
"""

def build_layers(layer_type: str, **kwargs):
    return LAYER_REGISTRY.get(layer_type)(**kwargs)
