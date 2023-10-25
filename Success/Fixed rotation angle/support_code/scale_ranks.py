from support_code.get_rank_cpu_0904 import Effective_Ranks, get_Effective_Ranks
import pandas as pd

def get_rank(train_loader):
  
  get_cifar10_gpu = get_Effective_Ranks(train_dataloader=train_loader)
  rk, Rk = get_cifar10_gpu.train_rk, get_cifar10_gpu.train_Rk
  rk_max_value = max(rk)  # 找到列表中的最大值
  rk_max_index = rk.index(rk_max_value)  # 找到最大值对应的索引
  Rk_value_max_rk_index = Rk[rk_max_index]

  return rk_max_index,rk_max_value, Rk_value_max_rk_index, rk[0]