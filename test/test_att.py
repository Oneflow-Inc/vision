import oneflow as flow
from flowvision.layers.attention import SEModule


def test_se():
    x = flow.randn(1, 48, 16, 16)
    se = SEModule(48)
    assert se(x).shape == x.shape


if __name__ == "__main__":
    test_se()
