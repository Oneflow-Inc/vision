import oneflow as flow

from flowvision.data import Mixup


def test_mixup(x, target, switch_prob=0.5, mode="batch"):
    mixup = Mixup(
        mixup_alpha=1.0,
        cutmix_alpha=1.0,
        switch_prob=switch_prob,
        label_smoothing=0.0,
        mode=mode,
    )
    x, target = mixup(x, target)
    return x, target


if __name__ == "__main__":
    x = flow.randn(16, 3, 224, 224).cuda()
    target = flow.arange(0, 16).cuda()
    test_mixup(x, target, mode="elem")
    test_mixup(x, target, mode="pair")
    test_mixup(x, target, mode="batch")
