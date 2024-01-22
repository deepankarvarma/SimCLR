import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from resent import get_resnet, name_to_params



class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()
        model, _ = get_resnet(depth=50, width=1, sk_ratio=0)
        print(model)
        self.net = []
        
        # encoder
        self.net = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))


    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
