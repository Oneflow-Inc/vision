from flowvision.utils import ModelEmaV2
from flowvision.models.alexnet import alexnet

if __name__ == "__main__":
    model = alexnet()
    model_ema = ModelEmaV2(model, decay=0.9999, device="cuda")
    new_model = alexnet()
    model_ema.update(new_model)
