import oneflow as flow
from oneflow.utils.data import DataLoader
import numpy as np
from PIL import Image
from flowvision.datasets import ImageFolder
from flowvision.data import RandomErasing
from flowvision.data import (
    rand_augment_transform,
    auto_augment_transform,
    augment_and_mix_transform,
)
from flowvision.data import create_transform


# class ImageNetDataLoader(DataLoader):
#     def __init__(
#         self, split="train", image_size=224, batch_size=16, num_workers=8
#     ):

#         transform = create_transform(
#             input_size=224,
#             is_training=True,
#             color_jitter=0.4,
#             auto_augment="rand-m9-mstd0.5-inc1",
#             re_prob=0.25,
#             re_mode="pixel",
#             re_count=1,
#             interpolation="bicubic"
#         )

#         self.dataset = ImageFolder(
#             root="/DATA/disk1/ImageNet/extract/train", transform=transform
#         )
#         super(ImageNetDataLoader, self).__init__(
#             dataset=self.dataset,
#             batch_size=batch_size,
#             shuffle=True if split == "train" else False,
#             num_workers=num_workers,
#         )


def test_random_erasing():
    x = flow.randn(4, 3, 224, 224)
    random_erase = RandomErasing(device="cpu")
    random_erase(x)


def test_aa():
    img_size = (224, 224)
    IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
    aa_params = dict(
        translate_const=int(min(img_size) * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
    )
    # test augmix
    x = np.random.randn(224, 224, 3)
    x = Image.fromarray(x, mode="RGB")
    augmix = augment_and_mix_transform(config_str="augmix-m5-w4-d2", hparams=aa_params)
    augmix(x)

    # test auto augmentation
    x = np.random.randn(224, 224, 3)
    x = Image.fromarray(x, mode="RGB")
    auto_augment = auto_augment_transform(
        config_str="original-mstd0.5", hparams=aa_params
    )
    auto_augment(x)

    # test rand augmentation
    x = np.random.randn(224, 224, 3)
    x = Image.fromarray(x, mode="RGB")
    rand_augment = rand_augment_transform(
        config_str="rand-m9-n3-mstd0.5", hparams=aa_params
    )
    rand_augment(x)


def test_transform_factory():
    x = np.random.randn(224, 224, 3)
    x = Image.fromarray(x, mode="RGB")
    transform = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        interpolation="bicubic",
    )
    transform(x)


if __name__ == "__main__":
    test_random_erasing()
    test_aa()
    test_transform_factory()
