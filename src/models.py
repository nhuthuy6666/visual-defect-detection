import torch
import torch.nn as nn
from torchvision import models

class DefectCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
def get_model(model_name="custom_cnn", num_classes=6, pretrained=True):
    model_name = model_name.lower()

    if model_name == "custom_cnn":
        print(f"Using custom CNN model (num_classes={num_classes})")
        return DefectCNN(num_classes=num_classes)

    elif model_name == "resnet18":
        print("Using ResNet18 model (pretrained={pretrained}, num_classes={num_classes})")
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "efficientnet_b0":
        print(f"Using EfficientNet-B0 model (pretrained={pretrained}, num_classes={num_classes})")
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported models are: 'custom_cnn', 'resnet18', 'efficientnet_b0'.")
    