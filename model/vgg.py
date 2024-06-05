"""
> VGG11
Total params: 9,231,114
Trainable params: 9,231,114
Non-trainable params: 0
Input size (MB): 0.01
Forward/backward pass size (MB): 3.71
Params size (MB): 35.21
Estimated Total Size (MB): 38.94

> VGG13
Total params: 9,416,010
Trainable params: 9,416,010
Non-trainable params: 0
Input size (MB): 0.01
Forward/backward pass size (MB): 5.96
Params size (MB): 35.92
Estimated Total Size (MB): 41.89

> VGG16
Total params: 14,728,266
Trainable params: 14,728,266
Non-trainable params: 0
Input size (MB): 0.01
Forward/backward pass size (MB): 6.57
Params size (MB): 56.18
Estimated Total Size (MB): 62.77

> VGG19
Total params: 20,040,522
Trainable params: 20,040,522
Non-trainable params: 0
Input size (MB): 0.01
Forward/backward pass size (MB): 7.18
Params size (MB): 76.45
Estimated Total Size (MB): 83.64
"""

import torch
import torch.nn as nn

# VGG16，有13层卷积层、3层全连接层
_cfgs = {
    # 数字为当前层输出的图像的通道数，字母'M'为最大池化层
    # VGG卷积层使用的全是3*3的卷积核、最大池化层使用的全是2*2的的卷积核、stride为2，因此每经过一层最大池化层，图片的宽高缩小为原来的一半
    # VGG16一共有5层最大池化层，输入的图片在到达全连接层时宽高缩小为原来的1/32
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_layer(cfg, batch_norm):
    """创建卷积层、池化层"""
    conv_layers = []
    in_channels = 3
    for layer in cfg:
        if layer == 'M':
            conv_layers.append(
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else:
            conv_layers.append(nn.Conv2d(
                in_channels=in_channels, out_channels=layer, kernel_size=3, padding=1
            ))
            # BatchNorm似乎是必须的，如果没有BatchNorm，训练误差、训练损失几乎没有变化
            # TODO：为什么会出现这种情况？BatchNorm的作用和原理。
            if batch_norm:
                conv_layers.append(nn.BatchNorm2d(layer))
            conv_layers.append(
                nn.ReLU(inplace=True)
            )
            in_channels = layer
    return nn.Sequential(*conv_layers)


class VGG(nn.Module):
    def __init__(self, model, num_classes=10, batch_norm=True):
        super(VGG, self).__init__()
        self.features = _make_layer(_cfgs[model], batch_norm)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))     # 限制全连接层收到的是一个宽高7*7的张量
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)

        # VGG16标准全连接层
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes)  # 输出层
        # )

        # 修改后的全连接层
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        VGG有5个池化层，因此当CIFAR10的3*32*32的图片到达全连接层前时，已经变成512*1*1，宽高缩小了2^5=32倍。
        此时只有两种选择:
        1. 更改VGG的全连接层，以满足512*1*1图片的需要
        2. VGG全连接层不变，扩大图片尺寸512*1*1至512*7*7
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_vgg11(num_classes=10, batch_norm=True):
    return VGG('vgg11', num_classes=num_classes, batch_norm=batch_norm)


def get_vgg13(num_classes=10, batch_norm=True):
    return VGG('vgg13', num_classes=num_classes, batch_norm=batch_norm)


def get_vgg16(num_classes=10, batch_norm=True):
    return VGG('vgg16', num_classes=num_classes, batch_norm=batch_norm)


def get_vgg19(num_classes=10, batch_norm=True):
    return VGG('vgg19', num_classes=num_classes, batch_norm=batch_norm)


# from torchsummary import summary
#
# summary(get_vgg19(10), input_size=(3, 32, 32), device='cpu')

