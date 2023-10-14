# 加载各种库
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR  # 实现cos函数式的变化
from torch.utils.tensorboard import SummaryWriter
from support_code.LeNet import LeNet
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_scale(
            # writer=None,  
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
            train_loader=None, test_loader=None, 
            # 可能需要的超参数
            momentum = 0.9,
            weight_decay = 0.0005,
            initial_lr = 0.01,
            num_epochs = 50,
            net = LeNet(),
            T_max = None,
            batch_size_train = 64,
            batch_size_test  = 64,
            min_angle = None,
            max_angle = None,
            scheduler=None,
            optimizer=None,
            path="scale",
            number=None):

  # print(device)
  
  # 如果目录不存在，创建它
  path = path + "/csv"
  if not os.path.exists(path):
      os.makedirs(path)

  net.to(device)  # 把数据转移到对应的device上,这里就是cuda

  if optimizer == None:
    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
  criterion = nn.CrossEntropyLoss()
  if scheduler == None:
    scheduler = CosineAnnealingLR(optimizer, T_max=int(num_epochs*len(train_loader)))
  
  # 建立字典来收集数据
  scale_dict = {
        "learning-rate": np.zeros(num_epochs),
        "train_loss": np.zeros(num_epochs),
        "train_acc": np.zeros(num_epochs),
        "test_loss": np.zeros(num_epochs),
        "test_acc": np.zeros(num_epochs)
    }

  for epoch in tqdm(range(num_epochs), unit="epoch", desc="Training"):
      # print('**************BEGIN EPOCH={}*************************'.format(epoch))
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
          scheduler.step()
          correct_num  += torch.sum(pred==target)

      train_loss = run_loss.item()/len(train_loader)
      train_acc = correct_num.item()/(len(train_loader)*batch_size_train)

      # print('epoch',epoch,'loss {:.2f}'.format(train_loss),'accuracy {:.2f}'.format(train_acc),'learning-rate {:.8f}'.format(optimizer.param_groups[0]['lr']))


      scale_dict["train_loss"][epoch] = train_loss
      scale_dict["train_acc"][epoch] = train_acc 
      scale_dict["learning-rate"][epoch] = optimizer.param_groups[0]['lr']

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
      # print('epoch',epoch,'loss {:.4f}'.format(test_loss),'accuracy {:.4f}'.format(test_acc))

      scale_dict["test_loss"][epoch] = test_loss
      scale_dict["test_acc"][epoch] = test_acc
    # 使用 Pandas 将 scale_dict 写入 CSV 文件
  df = pd.DataFrame(scale_dict)
  csv_filename = "{}-{:.4f}, {:.4f}.csv".format(
      number,min_angle, max_angle
  )
  df.to_csv(path + '/' + csv_filename, index=False)