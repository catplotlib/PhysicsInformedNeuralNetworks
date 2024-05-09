import torch
import torch.nn as nn

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.sin(self.linear(x))