from torchvision import models
import torch.nn as nn


def model_build(device, dense_neurons: int = 128, defrost=False, defrost_layers: int = 2):
    # Load the pretrained MobileNetV2 model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Congele todos os parâmetros inicialmente
    for param in model.features.parameters():
        param.requires_grad = False

    # Se `defrost=True`, descongele as últimas `defrost_layers` camadas do bloco `features`
    if defrost:
        for layer in list(model.features.children())[-defrost_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    # Modifique o classificador
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.last_channel, dense_neurons),
        nn.ReLU(),
        nn.Linear(dense_neurons, 1),
        nn.Sigmoid()
    )

    model = model.to(device)
    return model
