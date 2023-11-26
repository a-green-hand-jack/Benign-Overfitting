import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

"LeNet"
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out1 = out

        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out2 = out
        
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out3 = out
        out = F.relu(out)
        
        out = self.fc2(out)
        out4 = out
        out = F.relu(out)
        
        out = self.fc3(out)
        out5 = out
        return [x, out1, out2, out3, out4, out5]
    
    def children(self):
        # Return an iterator over child modules
        return iter([self.conv1, self.conv2, self.fc1, self.fc2, self.fc3])
    

''' MLP '''
class MLP(nn.Module):
    def __init__(self, channel=3, num_classes=10, im_size=(32, 32)):
        super(MLP, self).__init__()
        # print(im_size)
        self.fc_1 = nn.Linear(im_size[0] * im_size[1]*channel, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, num_classes)


    def forward(self, x):
        outs = []
        out = x.view(x.size(0), -1)
        
        out = F.relu(self.fc_1(out))
        out1 = out
        
        out = F.relu(self.fc_2(out))
        out2 = out
        out = self.fc_3(out)
        out3 = out
        return [x, out1, out2, out3]
    
    def children(self):
        # Return an iterator over child modules
        return iter([self.fc_1, self.fc_2, self.fc_3])

# -------------------定义自己的ResNet------------------------------
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
        
        return out

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

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet101():
    return ResNet(BasicBlock, [3, 4, 23, 3])

def ResNet152():
    return ResNet(BasicBlock, [3, 8, 36, 3])

#---------------------ResNet定义结束------------------------

class ModelInitializer:
    def __init__(self, model, seed=None):
        self.model = model  # 将模型设置为类成员变量
        if seed is not None:
            self.set_random_seed(seed)

        self.initialize_weights()
        # self.print_model_parameters()

    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def initialize_weights(self):
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.normal_(module.bias)  # 使用正态分布随机初始化偏置

    def print_model_parameters(self):
        for name, param in self.model.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}, sample:{param[:5]}")

    def get_layer_parameters(self, layer_index):
        layer_params = list(self.model.parameters())[layer_index]
        return layer_params


# 对模型参数进行高斯分布的随机初始化
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.linear import Linear

def init_weights(m: nn.Module) -> None:
    """
    初始化神经网络模型的权重和偏置（如果存在）。

    Args:
    - m (nn.Module): 需要初始化权重和偏置的神经网络模型

    Returns:
    - None
    """
    if isinstance(m, (_ConvNd, Linear)):  # 检查是否是卷积层或线性层
        init.normal_(m.weight.data, mean=0, std=0.01)  # 初始化权重为均值为0，标准差为0.01的正态分布
        if isinstance(m, _ConvNd) and m.bias is not None:
            init.constant_(m.bias.data, 0)  # 如果是卷积层且有偏置项，初始化偏置为常数0
        elif isinstance(m, Linear) and hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0)  # 如果是线性层且有偏置项，初始化偏置为常数0
