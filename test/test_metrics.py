import oneflow as flow
from flowvision.utils.metrics import accuracy
from flowvision.models.alexnet import alexnet


def test_acc(preds, target):
    return accuracy(preds, target, topk=(1, 5))


if __name__ == "__main__":
    target = flow.arange(0, 16)
    sample = flow.randn(16, 3, 224, 224)
    model = alexnet()
    preds = model(sample)
    top1_5 = test_acc(preds, target)
    print(top1_5)
