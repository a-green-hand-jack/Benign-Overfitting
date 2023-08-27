# 首先导入pytorch神经网络模块nn和函数式模块nn.functional
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self) -> None:
        super(MLP,self).__init__()
        channel=3
        num_classes=10
        im_size=(32, 32)
        self.fc_1 = nn.Linear(im_size[0] * im_size[1] * channel, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out
