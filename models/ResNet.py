# Based on the ResNet implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import math
import torch
from torch import nn
from torchvision.models.resnet import conv3x3

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_layer, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        
        # Adaptive padding for residual connection
        self.adaptive_pool = nn.AdaptiveAvgPool2d(None)

    def forward(self, x):
        identity = x
        
        residual = self.bn1(x)
        residual = self.relu1(residual)
        residual = self.conv1(residual)

        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)

        if self.downsample is not None:
            identity = self.downsample(identity)
            
        # Handle size mismatch by adaptively pooling the larger tensor to match the smaller one
        if residual.size() != identity.size():
            target_size = min(residual.size(2), identity.size(2)), min(residual.size(3), identity.size(3))
            residual = nn.functional.adaptive_avg_pool2d(residual, target_size)
            identity = nn.functional.adaptive_avg_pool2d(identity, target_size)
            
        return identity + residual

class Downsample(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(Downsample, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        assert nOut % nIn == 0
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)

class ResNetMouse(nn.Module):
    def __init__(self, depth, width=1, output_dim=2, channels=3, norm_layer=nn.BatchNorm2d):
        assert (depth - 2) % 6 == 0  # depth is 6N+2
        self.N = (depth - 2) // 6
        super(ResNetMouse, self).__init__()

        # First conv layer with adaptive padding
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, 16, kernel_size=7, stride=2, bias=False)
        )
        self.inplanes = 16
        self.layer1 = self._make_layer(norm_layer, 16 * width)
        self.layer2 = self._make_layer(norm_layer, 32 * width, stride=2)
        self.layer3 = self._make_layer(norm_layer, 64 * width, stride=2)
        self.bn = norm_layer(64 * width)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # This will handle any input size
        
        # Enhanced regression head for coordinate prediction using LayerNorm
        self.fc = nn.Sequential(
            nn.Linear(64 * width, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

        # Store original image dimensions for coordinate denormalization if needed
        self.original_h = 373
        self.original_w = 416

        # Initialization using He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(m, 'weight'):
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, norm_layer, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Downsample(self.inplanes, planes, stride)
        layers = [BasicBlock(self.inplanes, planes, norm_layer, stride, downsample)]
        self.inplanes = planes
        for i in range(self.N - 1):
            layers.append(BasicBlock(self.inplanes, planes, norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_unnormalized_coords(self, normalized_coords):
        """Convert normalized coordinates back to original image space"""
        x = normalized_coords[:, 0] * self.original_w
        y = normalized_coords[:, 1] * self.original_h
        return torch.stack([x, y], dim=1)

class ResNetCifar(nn.Module):
    def __init__(self, depth, width=1, classes=10, channels=3, norm_layer=nn.BatchNorm2d):
        assert (depth - 2) % 6 == 0         # depth is 6N+2
        self.N = (depth - 2) // 6
        super(ResNetCifar, self).__init__()

        # Following the Wide ResNet convention, we fix the very first convolution
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.inplanes = 16
        self.layer1 = self._make_layer(norm_layer, 16 * width)
        self.layer2 = self._make_layer(norm_layer, 32 * width, stride=2)
        self.layer3 = self._make_layer(norm_layer, 64 * width, stride=2)
        self.bn = norm_layer(64 * width)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * width, classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
    def _make_layer(self, norm_layer, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Downsample(self.inplanes, planes, stride)
        layers = [BasicBlock(self.inplanes, planes, norm_layer, stride, downsample)]
        self.inplanes = planes
        for i in range(self.N - 1):
            layers.append(BasicBlock(self.inplanes, planes, norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        