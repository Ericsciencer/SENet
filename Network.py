import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 1. 核心SE模块（论文核心） --------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # 压缩：全局平均池化 → (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 激励：全连接层学习通道注意力权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 升维
            nn.Sigmoid()  # 输出0~1的权重
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 压缩操作
        y = self.avg_pool(x).view(b, c)
        # 激励操作
        y = self.fc(y).view(b, c, 1, 1)
        # 缩放：通道加权（广播机制）
        return x * y.expand_as(x)

# -------------------------- 2. 带SE的残差块（适配ResNet） --------------------------
# 适用于 ResNet18/34
class SEBasicBlock(nn.Module):
    expansion = 1  # 通道扩展系数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        # 标准残差卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 嵌入SE模块
        self.se = SEBlock(out_channels, reduction)
        # 残差路径下采样（匹配维度）
        self.downsample = downsample

    def forward(self, x):
        residual = x
        # 主干卷积
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # SE通道注意力
        out = self.se(out)
        # 残差连接
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# 适用于 ResNet50/101/152
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        # 1x1降维 → 3x3卷积 → 1x1升维
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*4)
        self.relu = nn.ReLU(inplace=True)
        # 嵌入SE模块
        self.se = SEBlock(out_channels*4, reduction)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # SE通道注意力
        out = self.se(out)
        # 残差连接
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# -------------------------- 3. SE-ResNet 主网络 --------------------------
class SEResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, reduction=16):
        super(SEResNet, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.relu = nn.ReLU(inplace=True)

        # 构建4个残差层
        self.layer1 = self._make_layer(block, 64, layers[0], reduction)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, reduction)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, reduction)
        self.layer4 = self._make_layer(block, 512, layers[3], 2, reduction)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化（遵循官方ResNet）
        self._init_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1, reduction=16):
        """构建残差层"""
        downsample = None
        # 维度不匹配时，残差路径下采样
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels*block.expansion)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample, reduction)]
        self.in_channels = out_channels * block.expansion
        layers.extend([block(self.in_channels, out_channels, reduction=reduction) for _ in range(1, blocks)])
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始层
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# -------------------------- 4. 封装常用模型（直接调用） --------------------------
def se_resnet18(num_classes=1000):
    return SEResNet(SEBasicBlock, [2,2,2,2], num_classes)

def se_resnet34(num_classes=1000):
    return SEResNet(SEBasicBlock, [3,4,6,3], num_classes)

def se_resnet50(num_classes=1000):
    return SEResNet(SEBottleneck, [3,4,6,3], num_classes)

def se_resnet101(num_classes=1000):
    return SEResNet(SEBottleneck, [3,4,23,3], num_classes)

def se_resnet152(num_classes=1000):
    return SEResNet(SEBottleneck, [3,8,36,3], num_classes)

# -------------------------- 测试代码 --------------------------
if __name__ == '__main__':
    # 1. 创建SE-ResNet50（分类10类，适配自定义数据集）
    model = se_resnet50(num_classes=10)
    # 2. 随机输入：batch=2，3通道，224x224（ImageNet标准尺寸）
    dummy_input = torch.randn(2, 3, 224, 224)
    # 3. 前向推理
    output = model(dummy_input)
    # 4. 打印结果
    print(f"输入张量形状: {dummy_input.shape}")
    print(f"输出张量形状: {output.shape}")  # 输出: torch.Size([2, 10])