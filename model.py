from torchvision import models
import torch.nn as nn

def model_build(device, dense_neurons: int = 128):
    # Load the pretrained MobileNetV2 model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze all the parameters of the pretrained model
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Modify the classifier
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.last_channel, dense_neurons),
        nn.ReLU(),
        nn.Linear(dense_neurons, 1),
        nn.Sigmoid()
    )

    model = model.to(device)
    return model


