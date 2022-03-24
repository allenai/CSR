import torch.nn as nn
from src.models.backbones import FeatureLearner
from src.models.layers import *

class ResUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.feature_extractor = FeatureLearner(in_channels)
        self.up_head = ResUpHead(out_channels)

    def forward(self, x):
        self.feature_extractor(x)
        enc1, enc2, enc3, enc4, b = self.feature_extractor.intermediate_features

        return self.up_head(enc1, enc2, enc3, enc4, b)