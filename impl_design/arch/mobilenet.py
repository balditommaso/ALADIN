"""
Modified from: https://github.com/Xilinx/brevitas/blob/master/src/brevitas_examples/imagenet_classification/models/mobilenetv1.py
"""
import torch
from torch import nn, tensor
from typing import *


class MobileNetV1(nn.Module):

    def __init__(
        self,
        config: Tuple,
        in_channels: int = 3,
        num_classes: int = 10
    ) -> nn.Module:
        super(MobileNetV1, self).__init__()
        act = nn.ReLU
        
        self.pilot = self.bn_conv(3, 24, 2, act)

        features = []
        in_channels = 24
        for n, stride in config:
            features.append(
                self.dws_conv(in_channels, n * 24, stride, act)
            )
            in_channels = n * 24
            
        self.last_channel_out = in_channels
        
        features.append(nn.AdaptiveAvgPool2d((1,1)))
        self.features = nn.Sequential(*features)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(in_channels, num_classes)
        
        self._initialize_weights()
        
        
    @staticmethod
    def bn_conv(in_ch: int, out_ch: int, stride: tuple, act: type):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3,3), stride=stride, padding=(1,1), bias=False),
            nn.BatchNorm2d(out_ch),
            act(inplace=True)
        )


    @staticmethod
    def dws_conv(in_ch: int, out_ch: int, stride: tuple, act: type):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, (3,3), stride=stride, padding=(1,1), groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            act(inplace=True),
            nn.Conv2d(in_ch, out_ch, (1,1), stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(out_ch),
            act(inplace=True)
        )


    def forward(self, x: tensor):
        x = self.pilot(x)
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)


    def _initialize_weights(self, seed: int = -1):

        if seed >= 0:
            torch.manual_seed(seed)

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def mobilenet_v1(input_shape: List[int], num_classes: int) -> nn.Module:
    config = [
        (1, 1),   # 24 (dw)
        (2, 1),   # 48
        (2, 2),   # 48 dw, stride 2
        (4, 1),   # 96
        (4, 1),   # 96 dw
        (8, 2),   # 192
        (8, 1),   # 192 dw
        (16, 2),  # 384
        (16, 1),  # 384 dw
        (16, 1),  # 384
        (32, 1)   # 768 final projection
    ]

    net = MobileNetV1(
        config=config, 
        num_classes=num_classes
    )

    return net