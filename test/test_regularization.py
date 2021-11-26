import oneflow as flow
from flowvision.layers.regularization import StochasticDepth


def test_stochastic_depth(x, p=0.5, mode="row"):
    stochastic_depth = StochasticDepth(p=p, mode=mode)
    return stochastic_depth(x)


if __name__ == "__main__":
    x = flow.randn(16, 3, 48, 48)
    test_stochastic_depth(x)
