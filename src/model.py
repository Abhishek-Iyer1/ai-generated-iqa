import torch.nn as nn

from torchvision import models
from torchvision.models import ResNet


def load_model():

    # Load existing resnet model
    my_resnet: ResNet = models.resnet50(pretrained=True)

    # Freeze all weights
    for param in my_resnet.parameters():
        param.requires_grad = False

    # Change the last layer of the model and unfreeze it
    my_resnet.fc = nn.Linear(my_resnet.fc.in_features, 1)
    for param in my_resnet.fc.parameters():
        param.requires_grad = True

    print(my_resnet)
    # Return Model
    return my_resnet

if __name__ == "__main__":
    load_model()