import oneflow as flow
from flowvision.data import RandomErasing


if __name__ == "__main__":
    random_erase = RandomErasing(device="cpu")
    test_data = flow.randn(16, 3, 224, 224)
    random_erase(test_data)