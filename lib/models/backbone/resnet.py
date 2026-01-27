import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = ['resnet50', 'resnet152']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, last_stride=2, layers=[3, 4, 6, 3]):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=last_stride)

        self._init_weights()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * Bottleneck.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion

        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
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


def resnet50(last_stride=1, pretrained=True):
    model = ResNet(last_stride, [3, 4, 6, 3])
    if pretrained:
        _load_pretrained(model, model_urls['resnet50'])
    return model


def resnet152(last_stride=1, pretrained=True):
    model = ResNet(last_stride, [3, 8, 36, 3])
    if pretrained:
        _load_pretrained(model, model_urls['resnet152'])
    return model
