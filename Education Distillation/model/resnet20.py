import torch.nn as nn
import torch.nn.functional as F

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# 定义基本的ResNet块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 定义ResNet主体结构
class ResNet20(nn.Module):
    def __init__(self,num_classes = 33):
        super(ResNet20, self).__init__()
        self.in_planes = 16

        # 第一层：32x32输入，3x3卷积，16个输出通道，步长1，padding=1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 第二层：BasicBlock x 3，输入16，输出16，步长1
        self.layer1 = nn.Sequential(
            BasicBlock(self.in_planes, 16, stride=1),
            BasicBlock(16, 16, stride=1),
            BasicBlock(16, 16, stride=1)
        )

        # 第三层：BasicBlock x 3，输入16，输出32，步长2
        self.layer2 = nn.Sequential(

        )

        # 第四层：BasicBlock x 3，输入32，输出64，步长2
        self.layer3 = nn.Sequential(

        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(8),
            FlattenLayer(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc(out)
        return out

def resnet20(num_classes):
    return ResNet20(num_classes)

resnet20_layer2 = nn.Sequential(
    BasicBlock(16, 32, stride=2),
    BasicBlock(32, 32, stride=1),
    BasicBlock(32, 32, stride=1)
)
class Linear2(nn.Module):
    def __init__(self, num_classes):
        super(Linear2, self).__init__()
        self.num_classes = num_classes
        # Define the linear2 layer as an attribute
        self.linear2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((1, 1)),
            FlattenLayer(),  # Make sure FlattenLayer is defined elsewhere
            nn.Linear(64, self.num_classes)
        )
def resnet20_linear2(num_classes):
    return Linear2(num_classes)

resnet20_layer3 = nn.Sequential(
    BasicBlock(32, 64, stride=2),
    BasicBlock(64, 64, stride=1),
    BasicBlock(64, 64, stride=1)
)
class Linear3(nn.Module):
    def __init__(self, num_classes):
        super(Linear3, self).__init__()
        self.num_classes = num_classes
        # Define the linear2 layer as an attribute
        self.linear2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.AvgPool2d(8),
            FlattenLayer(),
            nn.Linear(64, num_classes)
        )

def resnet20_linear3(num_classes):
    return Linear3(num_classes)

