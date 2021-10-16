import flowvision.nn as nn
import oneflow as flow

from torchvision.models import resnet
# model = alexnet(pretrained=True)

# eval_flow_acc(model, "/DATA/disk1/ImageNet/extract", img_size=384, num_workers=0)

from flowvision.models import ModelCreator
# model = ModelCreator.create_model('vit_b_16_384', pretrained=True, model_dir="./test")
# print(model)
ModelCreator.model_table("*mobile*")

