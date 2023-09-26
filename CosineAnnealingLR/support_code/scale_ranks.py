from support_code.get_rank_cpu_0904 import Effective_Ranks, get_Effective_Ranks

def get_rank(scale, device, train_loader):
  get_cifar10_gpu = get_Effective_Ranks(train_dataloader=train_loader)
  rk, Rk = get_cifar10_gpu.train_rk, get_cifar10_gpu.train_Rk
  rk_max_value = max(rk)  # 找到列表中的最大值
  rk_max_index = rk.index(rk_max_value)  # 找到最大值对应的索引
  Rk_value_max_rk_index = Rk[rk_max_index]

  # 记录的标记
  # 构建动态字符串
  index_rk = "rk_max_index_scale"
  value_rk = "rk_max_value_scale"
  value_Rk_index_max_rk = "value_Rk_index_max_rk_scale" # 这里记录的是rk最大的时候的k对应的Rk的value
  r_0 = "r0_scale"

  # 建立字典，保存结果
  rank_dict = {index_rk:rk_max_index,
          value_rk:rk_max_value,
          value_Rk_index_max_rk:Rk_value_max_rk_index,
          r_0:rk[0]}
  return rk_max_index,rk_max_value, Rk_value_max_rk_index, rk[0]