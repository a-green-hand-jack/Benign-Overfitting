import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np
from get_rank_cpu import Effective_Ranks, get_Effective_Ranks, hello


# 以下代码为了在GPU上进行计算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 将get_Effective_Ranks类中的操作放在GPU上进行计算
class get_Effective_Ranks_GPU(get_Effective_Ranks):
    def __init__(self, dataset_name, path_to_dataset_folder, data_train_matrix=None, my_transform=transforms.Compose([ToTensor(),])):
        super(get_Effective_Ranks_GPU, self).__init__(dataset_name, path_to_dataset_folder, data_train_matrix, my_transform)
        # self.train_matrix = torch.tensor(self.train_matrix, dtype=torch.float32).to(device)
        '''
          ```
          UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).self.train_matrix = torch.tensor(self.train_matrix, dtype=torch.float32).to(device)
          ```

        这个警告是PyTorch提供的一个建议，用于在构造新的Tensor时避免一些潜在的问题。在你的代码中，你使用了`torch.tensor`来构造一个新的Tensor，而原始的`self.train_matrix`是一个已经存在的Tensor。PyTorch建议在构造新的Tensor时，应该使用`.clone().detach()`或者`.clone().detach().requires_grad_(True)`来从一个已有的Tensor创建新的Tensor，以避免一些潜在的问题。
        这里是警告的解释：
          1. `.clone()`: 这会创建原始Tensor的一个副本，包括数据和梯度信息。这意味着新的Tensor将与原始Tensor共享同样的数据和梯度。

          2. `.detach()`: 这将创建一个没有梯度信息的Tensor，意味着新的Tensor不会与原始Tensor共享梯度，即使原始Tensor有梯度。

          3. `.requires_grad_(True)`: 这会将Tensor的`requires_grad`属性设置为`True`，以便在后续的计算中可以计算梯度。
          在你的代码中，如果你想要从已有的`self.train_matrix`创建一个新的Tensor并将其移动到GPU上，可以使用以下方式：
              ```python
              self.train_matrix = self.train_matrix.clone().detach().to(device)
              ```
        这样，你将遵循PyTorch的建议，同时避免了警告消息。
        '''
        self.train_matrix = self.train_matrix.clone().detach().to(device)

    
    def convert_to_matrix(self):
        # 不再将数据转换为numpy数组，而是将其转换为GPU上的torch.tensor
        train_vectors = [] 
        for images, labels in self.train_loader:
            images = images.view(images.size(0), -1)
            images = images.to(device)  # 将数据转移到GPU上
            train_vectors.append(images[0])
        self.train_matrix = torch.stack(train_vectors)

