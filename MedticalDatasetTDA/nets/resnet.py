import torch.nn as nn
import torch.nn.functional as F
import torch

# 基本块 BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 瓶颈块 Bottleneck
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ResNet类
# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
#         self.bn1 = nn.BatchNorm2d(64).to(device)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes).to(device)

#     def _make_layer(self, block, planes, num_blocks, stride,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride).to(device))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out1 = out
#         out = self.layer2(out)
#         out2 = out
#         out = self.layer3(out)
#         out3 = out
#         out = self.layer4(out)
#         out4 = out
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return x, out1, out2, out3, out4, out

    
#     def children(self):
#         # Return an iterator over child modules
#         return iter([self.layer1, self.layer2, self.layer3, self.layer4, self.linear])
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 这里保证了self.conv1和self.bn1的参数在设备上
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        self.bn1 = nn.BatchNorm2d(64).to(device)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes).to(device)

    def _make_layer(self, block, planes, num_blocks, stride,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layer = block(self.in_planes, planes, stride)
            layer = layer.to(device)  # 将子模块移动到设备上
            layers.append(layer)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out1 = out
        out = F.relu(out)

        out = self.layer2(out)
        out2 = out
        out = F.relu(out)

        out = self.layer3(out)
        out3 = out
        out = F.relu(out)

        out = self.layer4(out)
        out4 = out
        out = F.relu(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return x, out1, out2, out3, out4, out
    
    def children(self):
        # Return an iterator over child modules
        return iter([self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool])

# 创建ResNet不同深度的函数
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
