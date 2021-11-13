import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """NLL Loss with label smoothing
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x: flow.Tensor, target: flow.Tensor) -> flow.Tensor:
        # TODO: register F.log_softmax() function and switch flow.log(flow.softmax()) to F.log_softmax()
        logprobs = flow.log_softmax(x, dim=-1)
        # TODO: fix gather bug when dim < 0
        # FIXME: only support cls task now
        nll_loss = -logprobs.gather(dim=1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: flow.Tensor, target: flow.Tensor) -> flow.Tensor:
        loss = flow.sum(-target * flow.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
