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
