import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):  # y = x + F(x)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:  # 下采样保证了x和F(x)形状相同
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def _make_layer(inplanes, planes, stride=1):
    return nn.Sequential(
        ResidualBlock(inplanes, planes, stride=stride),
        ResidualBlock(planes, planes, stride=1)
    )


# 输入图像大小3*32*32，10分类任务
class ResNet18(nn.Module):

    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = _make_layer(64, 64, stride=1)
        self.conv3_x = _make_layer(64, 128, stride=2)
        self.conv4_x = _make_layer(128, 256, stride=2)
        self.conv5_x = _make_layer(256, 512, stride=2)
        # 对于大小为3*32*32的输入图片，到达这里时，图片变成了512*1*1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_resnet18(num_classes=10):
    return ResNet18(num_classes=num_classes)

# from torchsummary import summary
#
# summary(ResNet18(), (3, 32, 32), device='cpu')

"""
================================================================
Total params: 11,183,562
Trainable params: 11,183,562
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.29
Params size (MB): 42.66
Estimated Total Size (MB): 43.96
----------------------------------------------------------------
"""