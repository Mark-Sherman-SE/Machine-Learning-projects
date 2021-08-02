import torch
from model import *

models = (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152
)


def test():
    for model in models:
        model = model(img_channels=3, num_classes=1000)
        y = model(torch.randn(4, 3, 224, 224))
        assert y.detach().numpy().shape == (4, 1000), "Output must be a tensor with size (4, 1000)"
    print("All outputs are correct")


if __name__ == "__main__":
    test()
