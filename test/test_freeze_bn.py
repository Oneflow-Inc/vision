from flowvision.utils import freeze_batch_norm_2d, unfreeze_batch_norm_2d
from flowvision.models.resnet import resnet50

if __name__ == "__main__":
    model = resnet50()
    freeze_batch_norm_2d(model)
    for name, module in model.named_modules():
        print(type(module))
    
    unfreeze_batch_norm_2d(model)
    for name, module in model.named_modules():
        print(type(module))