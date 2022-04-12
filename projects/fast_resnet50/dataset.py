import os

from flowvision import datasets, transforms
from flowvision.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from flowvision.transforms.functional import str_to_interp_mode
from flowvision.data import Mixup


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == "imagenet":
        prefix = "train" if is_train else "val"
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)



def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        t = []
        # this should always dispatch to transforms_imagenet_train
        t.append(transforms.RandomResizedCrop(
            size=(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
            interpolation=str_to_interp_mode(config.DATA.INTERPOLATION)
        ))
        t.append(transforms.RandomHorizontalFlip(p=0.5))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

        return transforms.Compose(t)

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(
                    size, interpolation=str_to_interp_mode(config.DATA.INTERPOLATION)
                ),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=str_to_interp_mode(config.DATA.INTERPOLATION),
                )
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)