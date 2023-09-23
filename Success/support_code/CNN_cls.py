import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
class CNN_cls(nn.Module):
    def __init__(self,in_dim):
        super(CNN_cls,self).__init__()
        self.conv1 = nn.Conv2d(in_dim,32,1,1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,1,1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64,128,1,1)
        self.lin1 = nn.Linear(128*8*8,512)
        self.lin2 = nn.Linear(512,64)
        self.lin3 = nn.Linear(64,10)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(-1,128*8*8)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        return x
