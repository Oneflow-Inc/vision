from flowvision.models import vit_l_16_384
from one_utils import eval_flow_acc


model = vit_l_16_384(pretrained=True)

eval_flow_acc(model, "/DATA/disk1/ImageNet/extract", img_size=384, num_workers=0)
