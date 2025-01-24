import os

from oneflow.utils.data import DataLoader

from flowvision import datasets, transforms
from flowvision.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from flowvision.transforms.functional import str_to_interp_mode


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(
        is_train=True, config=config
    )
    config.freeze()
    dataset_val, _ = build_dataset(is_train=False, config=config)
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        drop_last=True,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        drop_last=False,
    )
    return dataset_train, dataset_val, data_loader_train, data_loader_val


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == "imagenet":
        prefix = "train" if is_train else "val"
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == "cifar100":
        dataset = datasets.CIFAR100(
            root = config.DATA.DATA_PATH,
            train=is_train,
            transform=transform,
            download=True,
        )
        nb_classes = 100
    else:
        raise NotImplementedError("We only support ImageNet and CIFAR100 Now.")

    return dataset, nb_classes


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