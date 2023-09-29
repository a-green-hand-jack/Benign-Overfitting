import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from support_code.LeNet import LeNet

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
    
# 设置设备和种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
batch_size_train = 64
batch_size_test  = 64
epochs = 50
learning_rate = 0.01
momentum = 0.5
net = LeNet()
# net = CNN_cls(3)
net.to(device)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.5,), (0.5,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.5,), (0.5,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

optimizer = optim.SGD(net.parameters(), lr=learning_rate,momentum=momentum)
criterion = nn.CrossEntropyLoss()

print("****************Begin Training****************")
net.train()
for epoch in range(epochs):
    run_loss = 0
    correct_num = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)   # 把数据转移到对应的device上,这里就是cuda
        out = net(data)
        _,pred = torch.max(out,dim=1)
        optimizer.zero_grad()
        loss = criterion(out,target)
        loss.backward()
        run_loss += loss
        optimizer.step()
        correct_num  += torch.sum(pred==target)
    print('epoch',epoch,'loss {:.2f}'.format(run_loss.item()/len(train_loader)),'accuracy {:.2f}'.format(correct_num.item()/(len(train_loader)*batch_size_train)))



    print("****************Begin Testing****************")
    net.eval()
    test_loss = 0
    test_correct_num = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)   # 把数据转移到对应的device上,这里就是cuda
        out = net(data)
        _,pred = torch.max(out,dim=1)
        test_loss += criterion(out,target)
        test_correct_num  += torch.sum(pred==target)
    print('loss {:.2f}'.format(test_loss.item()/len(test_loader)),'accuracy {:.2f}'.format(test_correct_num.item()/(len(test_loader)*batch_size_test)))
