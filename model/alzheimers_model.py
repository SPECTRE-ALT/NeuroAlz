"""
This code defines a custom neural network called AlzheimerNet. It uses a pre-trained EfficientNet-B0 model as the base and modifies the classifier layer to output the desired number of classes (4 in this case, for different stages of Alzheimer's). The forward method defines how input data flows through the network
"""

import torch.nn as nn
from torchvision import models

class AlzheimerNet(nn.Module):
    def __init__(self, num_classes=4, sophisticated=True):
        super(AlzheimerNet, self).__init__()
        # Load pre-trained EfficientNetB0
        # Using weights instead of pretrained to avoid warnings if possible, fallback for older torch versions
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.base_model = models.efficientnet_b0(weights=weights)
        except AttributeError:
            self.base_model = models.efficientnet_b0(pretrained=True)
        
        # Get input features of the final layer
        # EfficientNet-B0 classifier is typically:
        # (classifier): Sequential(
        #   (0): Dropout(p=0.2, inplace=True)
        #   (1): Linear(...)
        # )
        
        num_features = self.base_model.classifier[1].in_features
        
        if sophisticated:
            # Upgrade to a modern, robust classifier head
            self.base_model.classifier = nn.Sequential(
                nn.LayerNorm(num_features), # Better for medical scans than BatchNorm
                nn.Linear(num_features, 512),
                nn.Mish(), # Mish activation is more accurate than ReLU for deep medical networks
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.Mish(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        else:
            # Legacy/Simple architecture (Single Linear Layer)
            self.base_model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
