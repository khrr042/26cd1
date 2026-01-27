import math
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = [
    'ResNet_IBN',
    'resnet50_ibn_a',
    'resnet101_ibn_a',
    'resnet152_ibn_a',
]

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class IBN(nn.Module):
    def __init__(self, planes):
        super().__init__()
        half = planes // 2
        self.half_channel = half
        self.IN = nn.InstanceNorm2d(half, affine=True)
        self.BN = nn.BatchNorm2d(planes - half)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.half_channel, x.size(1) - self.half_channel], dim=1)
        return torch.cat((self.IN(x1), self.BN(x2)), dim=1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = IBN(planes) if ibn else nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = IBN(planes) if ibn else nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.se = SELayer(planes * self.expansion, reduction)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class ResNet_IBN(nn.Module):
    def __init__(self, last_stride, block, layers):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn = planes != 512
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def load_param(self, model_path):
        try:
            param_dict = torch.load(model_path, map_location='cpu')
        except Exception as e:
            print(f"Error loading checkpoint from {model_path}: {e}")
            return

        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
            
        print(f"Loading pretrained model from {model_path}...")
        
        model_dict = self.state_dict()
        new_state_dict = {k: v for k, v in param_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        
        if len(new_state_dict) == 0:
             new_state_dict = {k.replace('module.', ''): v for k, v in param_dict.items() if k.replace('module.', '') in model_dict and v.size() == model_dict[k.replace('module.', '')].size()}
        
        if len(new_state_dict) > 0:
            model_dict.update(new_state_dict)
            self.load_state_dict(model_dict)
            print(f"Successfully loaded {len(new_state_dict)} layers.")
        else:
            print("Warning: No matching layers found in the checkpoint.")


def _load_pretrained(model, url):
    state_dict = load_state_dict_from_url(url, progress=True)
    model.load_state_dict(state_dict, strict=False)

def resnet50_ibn_a(last_stride=1, pretrained=False):
    model = ResNet_IBN(last_stride, Bottleneck_IBN, [3, 4, 6, 3])
    if pretrained:
        _load_pretrained(model, model_urls['resnet50'])
    return model


def resnet101_ibn_a(last_stride=1, pretrained=False):
    model = ResNet_IBN(last_stride, Bottleneck_IBN, [3, 4, 23, 3])
    if pretrained:
        _load_pretrained(model, model_urls['resnet101'])
    return model

def resnet152_ibn_a(last_stride=1, pretrained=False):
    model = ResNet_IBN(last_stride, Bottleneck_IBN, [3, 8, 36, 3])
    if pretrained:
        _load_pretrained(model, model_urls['resnet152'])
    return model


def se_resnet101_ibn_a(last_stride=1):
    return ResNet_IBN(last_stride, SEBottleneck, [3, 4, 23, 3])