import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnext50_32x4d


class Model(nn.Module):
    def __init__(self, ensemble_size, feature_dim, model_type, share_type):
        super(Model, self).__init__()

        # backbone
        backbones = {'resnet18': (resnet18, 1), 'resnet34': (resnet34, 1), 'resnet50': (resnet50, 4),
                     'resnext50_32x4d': (resnext50_32x4d, 4)}
        backbone, expansion = backbones[model_type]
        module_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']

        # configs
        self.ensemble_size, self.feature_dim = ensemble_size, feature_dim
        if share_type == 'none' or module_names.index(share_type) <= 2:
            self.stride = 16
        elif module_names.index(share_type) in [3, 4]:
            self.stride = 8
        elif module_names.index(share_type) >= 5:
            self.stride = 8 / (2 ** (module_names.index(share_type) - 4))

        # common features
        self.common_extractor = []
        common_module_names = [] if share_type == 'none' else module_names[:module_names.index(share_type) + 1]
        for name, module in backbone().named_children():
            if name in common_module_names:
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.common_extractor.append(module)
        self.common_extractor = nn.Sequential(*self.common_extractor)
        print("# trainable common feature parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in self.common_extractor.parameters()))

        # branch attention
        # TODO: How to init attention
        self.branch_att = nn.Parameter(torch.rand(ensemble_size, self.stride * 2, self.stride * 2), requires_grad=True)
        std = 1. / math.sqrt((self.stride * 2) ** 2 / 3)
        nn.init.uniform_(self.branch_att, a=-std, b=std)

        # individual features
        self.head, self.layer1, self.layer2, self.layer3, self.layer4 = [], [], [], [], []
        individual_module_names = module_names if share_type == 'none' else \
            module_names[module_names.index(share_type) + 1:]
        for i in range(ensemble_size):
            heads = []
            for name, module in backbone().named_children():
                if name in individual_module_names and name in ['conv1', 'bn1', 'relu', 'maxpool']:
                    if name == 'conv1':
                        module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                    heads.append(module)
                if name in individual_module_names and name == 'layer1':
                    self.layer1.append(module)
                if name in individual_module_names and name == 'layer2':
                    self.layer2.append(module)
                if name in individual_module_names and name == 'layer3':
                    self.layer3.append(module)
                if name in individual_module_names and name == 'layer4':
                    self.layer4.append(module)
            self.head.append(nn.Sequential(*heads))
        self.head = nn.ModuleList(self.head)
        self.layer1 = nn.ModuleList(self.layer1)
        self.layer2 = nn.ModuleList(self.layer2)
        self.layer3 = nn.ModuleList(self.layer3)
        self.layer4 = nn.ModuleList(self.layer4)
        print("# trainable individual feature parameters:",
              (sum(param.numel() if param.requires_grad else 0 for param in self.head.parameters()) +
               sum(param.numel() if param.requires_grad else 0 for param in self.layer1.parameters()) + sum(
                          param.numel() if param.requires_grad else 0 for param in self.layer2.parameters()) + sum(
                          param.numel() if param.requires_grad else 0 for param in self.layer3.parameters()) + sum(
                          param.numel() if param.requires_grad else 0 for param in
                          self.layer4.parameters())) // ensemble_size)

        # global features
        self.global_extractors = nn.ModuleList([nn.Linear(512 * expansion, feature_dim) for _ in range(ensemble_size)])
        print("# trainable global feature parameters:", sum(param.numel() if param.requires_grad else 0 for param in
                                                            self.global_extractors.parameters()) // ensemble_size)

    def forward(self, x):
        batch_size = x.size(0)
        common_feature = self.common_extractor(x)
        common_feature = F.interpolate(common_feature, size=self.stride * 2, mode='bilinear', align_corners=True)
        branch_weight = torch.sigmoid(self.branch_att).unsqueeze(dim=0)

        out = []
        for i in range(self.ensemble_size):
            individual_feature = branch_weight[:, i, :, :].unsqueeze(dim=1) * common_feature
            if len(self.head) != 0:
                individual_feature = self.head[i](individual_feature)
            if len(self.layer1) != 0:
                individual_feature = self.layer1[i](individual_feature)
            if len(self.layer2) != 0:
                individual_feature = self.layer2[i](individual_feature)
            if len(self.layer3) != 0:
                individual_feature = self.layer3[i](individual_feature)
            if len(self.layer4) != 0:
                individual_feature = self.layer4[i](individual_feature)
            global_feature = F.adaptive_avg_pool2d(individual_feature, output_size=(1, 1)).view(batch_size, -1)
            global_feature = self.global_extractors[i](global_feature)
            out.append(global_feature)
        # [B, E, D]
        out = F.normalize(torch.stack(out, dim=1), dim=-1)
        return out
