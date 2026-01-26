# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone
from .layers.pooling import GeM 

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class FC(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(FC, self).__init__()
        self.fc = nn.Linear(inplanes, outplanes)
        self.bn = nn.BatchNorm1d(outplanes)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.fc(x)
        return self.act(x)

class GDN(nn.Module):
    def __init__(self, inplanes, outplanes, intermediate_dim=256):
        super(GDN, self).__init__()
        self.fc1 = FC(inplanes, intermediate_dim)
        self.fc2 = FC(intermediate_dim, outplanes)

    def forward(self, x):
        intermediate = self.fc1(x)
        out = self.fc2(intermediate)
        return intermediate, torch.softmax(out, dim=1)

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path,
                 neck, neck_feat, model_name, pretrain_choice, cfg):
        super().__init__()

        self.base = build_backbone(model_name, last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        if cfg.MODEL.POOLING_METHOD == 'GeM':
            self.gap = GeM()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

def forward(self, x, label=None, return_featmap=False):
    featmap = self.base(x)
    if return_featmap:
        return featmap

    global_feat = self.gap(featmap)
    global_feat = global_feat.flatten(1)

    feat = self.bottleneck(global_feat)

    if self.training:
        cls_score = self.classifier(feat)
        return cls_score, feat
    else:
        return feat if self.neck_feat == 'after' else global_feat
