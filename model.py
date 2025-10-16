import timm
import torch
import torch.nn as nn


class MaxViTModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = timm.create_model("maxvit_tiny_tf_224", pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
