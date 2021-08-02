import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(intermediate_channels * self.expansion)
        )
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels=3, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)

        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flattening = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = self.flattening(x)
        x = self.fc(x)

        return x

    def _make_layer(self, block, blocks_num, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels * block.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(intermediate_channels * block.expansion),
            )

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * block.expansion

        for i in range(blocks_num - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def resnet18(img_channels=3, num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], img_channels, num_classes)


def resnet34(img_channels=3, num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], img_channels, num_classes)


def resnet50(img_channels=3, num_classes=1000):
    return ResNet(BottleNeck, [3, 4, 6, 3], img_channels, num_classes)


def resnet101(img_channels=3, num_classes=1000):
    return ResNet(BottleNeck, [3, 4, 23, 3], img_channels, num_classes)


def resnet152(img_channels=3, num_classes=1000):
    return ResNet(BottleNeck, [3, 8, 36, 3], img_channels, num_classes)
