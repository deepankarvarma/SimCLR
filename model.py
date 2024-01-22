import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, feature_dim=128, pretrain_path=None):
        super(Model, self).__init__()

        # Load pre-trained ResNet weights
        if pretrain_path is not None:
            checkpoint = torch.load(pretrain_path)
            resnet_weights = checkpoint['net']  # Adjust this based on your checkpoint structure
            self.load_state_dict(resnet_weights)

        # Define your encoder
        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        
        # Encoder
        self.f = nn.Sequential(*self.f)
        
        # Projection head
        self.g = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

# class Model(nn.Module):
#     def __init__(self, feature_dim=128):
#         super(Model, self).__init__()

#         self.f = []
#         for name, module in resnet50().named_children():
#             if name == 'conv1':
#                 module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#             if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
#                 self.f.append(module)
#         # encoder
#         self.f = nn.Sequential(*self.f)
#         # projection head
#         self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
#                                nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))


#     def forward(self, x):
#         x = self.f(x)
#         feature = torch.flatten(x, start_dim=1)
#         out = self.g(feature)
#         return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


#     def forward(self, x):
#         x = self.f(x)
#         feature = torch.flatten(x, start_dim=1)
#         out = self.g(feature)
#         return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
