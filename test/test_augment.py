import oneflow as flow
import numpy as np
from PIL import Image
from flowvision.transforms import RandomErasing
from flowvision.transforms import rand_augment_transform, auto_augment_transform, augment_and_mix_transform



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
    auto_augment = auto_augment_transform(config_str="original-mstd0.5", hparams=aa_params)
    auto_augment(x)

    # test rand augmentation
    x = np.random.randn(224, 224, 3)
    x = Image.fromarray(x, mode="RGB")
    rand_augment = rand_augment_transform(config_str="rand-m9-n3-mstd0.5", hparams=aa_params)
    rand_augment(x)



if __name__ == "__main__":
    test_random_erasing()
    test_aa()