from flowvision.models import ModelCreator


def build_model(config):
    model_arch = config.MODEL.ARCH
    model = ModelCreator.create_model(model_arch, pretrained=config.MODEL.PRETRAINED)
    return model
