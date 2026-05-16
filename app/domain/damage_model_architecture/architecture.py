import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import torch.nn.functional as F
class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        resnet = models.resnet34(weights='DEFAULT')
        # local_weights = torch.load('app/weights/resnet34-b627a593.pth',map_location='cpu')
        # resnet.load_state_dict(local_weights)

        if in_channels != 3:
            old_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                nn.init.zeros_(resnet.conv1.weight)
                resnet.conv1.weight[:, :3, :, :] = old_conv.weight

        self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # /2
        self.maxpool = resnet.maxpool                                    # /4
        self.layer1 = resnet.layer1                                      # /4
        self.layer2 = resnet.layer2                                      # /8
        self.layer3 = resnet.layer3                                      # /16
        self.layer4 = resnet.layer4                                      # /32

    def forward(self, x):
        x0 = self.base(x)       # 64 channels
        x1 = self.layer1(self.maxpool(x0)) # 64 channels
        x2 = self.layer2(x1)    # 128 channels
        x3 = self.layer3(x2)    # 256 channels
        x4 = self.layer4(x3)    # 512 channels
        return [x0, x1, x2, x3, x4]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_p=0.2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = torch.nn.functional.interpolate(x, size=skip.shape[2:])

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DamageModel(nn.Module):
    def __init__(self, mode='pre_post', num_classes=5):
        super(DamageModel, self).__init__()
        self.mode = mode
        if mode in ['pre_post', 'pre_post_premask']:
            f = 2
        else:
            f = 1

        in_ch = 4 if mode in ['post_premask', 'pre_post_premask'] else 3
        self.encoder = ResNet34Encoder(in_channels=in_ch)

        self.up4 = DecoderBlock(512*f, 256*f, 256, dropout_p=0.3)
        self.up3 = DecoderBlock(256, 128*f, 128, dropout_p=0.2)
        self.up2 = DecoderBlock(128, 64*f, 64, dropout_p=0.1)
        self.up1 = DecoderBlock(64, 64*f, 64, dropout_p=0.0)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, *inputs):

        if self.mode == 'post_only':
            feat = self.encoder(inputs[0])
            x = feat[4]
            skips = [feat[3], feat[2], feat[1], feat[0]]

        elif self.mode == 'post_premask':
            combined = torch.cat((inputs[0], inputs[1]), dim=1)
            feat = self.encoder(combined)
            x = feat[4]
            skips = [feat[3], feat[2], feat[1], feat[0]]

        elif self.mode == 'pre_post':
            f_pre, f_post = self.encoder(inputs[0]), self.encoder(inputs[1])
            x = torch.cat([f_pre[4], f_post[4]], dim=1)
            skips = [torch.cat([f_pre[i], f_post[i]], dim=1) for i in range(3, -1, -1)]

        elif self.mode == 'pre_post_premask':
            pre_f = self.encoder(torch.cat((inputs[0], inputs[2]), dim=1))
            post_f = self.encoder(torch.cat((inputs[1], inputs[2]), dim=1))
            x = torch.cat([pre_f[4], post_f[4]], dim=1)
            skips = [torch.cat([pre_f[i], post_f[i]], dim=1) for i in range(3, -1, -1)]

        x = self.up4(x, skips[0])
        x = self.up3(x, skips[1])
        x = self.up2(x, skips[2])
        x = self.up1(x, skips[3])

        #normalizing features for the ldam loss
        features = self.final_up(x)
        features = F.normalize(features, p=2, dim=1)
        norm_weight = F.normalize(self.final_conv.weight, p=2, dim=1)
        logits = F.conv2d(features, norm_weight, bias=None)

        return logits, features
        #return self.final_conv(self.final_up(x))