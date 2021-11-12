from flowvision.layers.layer_registry import Layer_Registry
ATT_REGISTRY = Layer_Registry("attention")
ATT_REGISTRY.__doc__ = """
Registry for attention modules, which be used as a plug-and-play block in neural network
"""

def build_attention(attn_type: str):
    """
    Build an attention module from attn_type
    """
    pass
