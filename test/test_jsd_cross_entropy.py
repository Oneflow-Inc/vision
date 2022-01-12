import oneflow as flow
import oneflow.nn as nn

from flowvision.loss import JsdCrossEntropy


if __name__ == "__main__":
    output = flow.randn(48, 1000)
    target = flow.arange(0, 47)
    jsd_loss = JsdCrossEntropy(num_splits=3, alpha=12, smoothing=0.1)
    jsd_loss(output, target)