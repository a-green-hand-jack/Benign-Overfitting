import torch.nn as nn
import torch.nn.functional as F

''' MLP '''
class MLP(nn.Module):
    def __init__(self, channel=3, num_classes=10, im_size=(32, 32)):
        super(MLP, self).__init__()
        # print(im_size)
        self.fc_1 = nn.Linear(im_size[0] * im_size[1]*channel, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out