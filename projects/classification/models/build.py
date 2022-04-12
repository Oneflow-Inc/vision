from flowvision.models import ModelCreator
from .fast_resnet50 import fast_resnet50


def build_model(config):
    model_arch = config.MODEL.ARCH
    if model_arch == "fast_resnet50":
        model = fast_resnet50()
    else:
        model = ModelCreator.create_model(model_arch, pretrained=config.MODEL.PRETRAINED)
    return model
