# 加载各种库
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR  # 实现cos函数式的变化
from torch.utils.tensorboard import SummaryWriter
from support_code.LeNet import LeNet

def get_scale(
            writer=None,  
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
            train_loader=None, valid_loader=None, test_loader=None, 
            # 可能需要的超参数
            momentum = 0.9,
            weight_decay = 0.0005,
            initial_lr = 0.01,
            num_epochs = 50,
            folder_path = "LeNet-0921",
            net = LeNet(),
            T_max = None,
            batch_size_train = 64,
            batch_size_test  = 64,
            train_loss_name = "train_loss",
            train_acc_name = "train_acc",
            test_loss_name = "test_loss",
            test_acc_name = "test_acc"):

  print(device)
  # flog = True

  writer = SummaryWriter(folder_path)

  net.to(device)  # 把数据转移到对应的device上,这里就是cuda

  optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(num_epochs):
      print('**************BEGIN EPOCH={}*************************'.format(epoch))
    #   print("****************Begin Training****************")
      net.train()
      run_loss = 0
      correct_num = 0
      for batch_idx, (data, target) in enumerate(train_loader):
          data, target = data.to(device), target.to(device)   # 把数据转移到对应的device上,这里就是cuda
          # if flog == True:
          #   print(data.shape)
          #   flog = False
          out = net(data)
          _,pred = torch.max(out,dim=1)
          optimizer.zero_grad()
          loss = criterion(out,target)
          loss.backward()
          run_loss += loss
          optimizer.step()
          correct_num  += torch.sum(pred==target)

      train_loss = run_loss.item()/len(train_loader)
      train_acc = correct_num.item()/(len(train_loader)*batch_size_train)
      print('epoch',epoch,'loss {:.2f}'.format(train_loss),'accuracy {:.2f}'.format(train_acc))
      writer.add_scalar(train_loss_name, train_loss, epoch)
      writer.add_scalar(train_acc_name, train_acc, epoch)
          # 更新模型权重并记录到Tensorboard
      for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    #   print("****************Begin Testing****************")
      net.eval()
      test_loss = 0
      test_correct_num = 0
      for batch_idx, (data, target) in enumerate(test_loader):
          data, target = data.to(device), target.to(device)   # 把数据转移到对应的device上,这里就是cuda
          out = net(data)
          _,pred = torch.max(out,dim=1)
          test_loss += criterion(out,target)
          test_correct_num  += torch.sum(pred==target)    

      test_loss = test_loss.item()/len(test_loader)
      test_acc = test_correct_num.item()/(len(test_loader)*batch_size_test)
      print('epoch',epoch,'loss {:.2f}'.format(test_loss),'accuracy {:.2f}'.format(test_acc))
      writer.add_scalar(test_loss_name, test_loss, epoch)
      writer.add_scalar(test_acc_name, test_acc, epoch)
