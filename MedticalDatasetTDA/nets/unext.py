# 只是放在这里，不一定能用
import torch
import torchvision
from torchvision.utils import save_image
import torch.nn.functional as F
import os

__all__ = ['UNext']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn, Tensor
from torch.nn.init import kaiming_normal_, constant_
import math
from typing import Tuple
import torch.nn as nn


class DWConv(nn.Module):
    def __init__(self, dim: int = 768):
        '''
        DWConv模块，用于实现深度卷积操作。

        参数：
        - dim：输入张量的通道数（默认为768）。
        '''
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        '''
        DWConv前向传播方法。

        参数：
        - x：输入张量，shape为[B, N, C]，其中B为批次大小，N为序列长度，C为通道数。
        - H：图像的高度。
        - W：图像的宽度。

        返回：
        - x：输出张量，shape为[B, N, C_out]，其中C_out为输出通道数。

        注意：输入张量x的shape中的N要与H * W相等。
        '''
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x





class shiftmlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None,
                 act_layer: nn.Module = nn.GELU, drop: float = 0., shift_size: int = 5):
        '''
        ShiftMLP模块，一个基于移位操作的多层感知机（MLP）模块。

        参数：
        - in_features：输入特征的维度。
        - hidden_features：隐藏层特征的维度（默认为None，等于输入特征维度）。
        - out_features：输出特征的维度（默认为None，等于输入特征维度）。
        - act_layer：激活函数（默认为nn.GELU）。
        - drop：dropout 比例（默认为0）。
        - shift_size：移位大小（默认为5）。
        '''
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        '''
        初始化模型的权重。

        参数：
        - m：网络模块中的子模块。
        '''
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        '''
        ShiftMLP前向传播方法。

        参数：
        - x：输入张量，shape为[B, N, C]，其中B为批次大小，N为序列长度，C为通道数。
        - H：图像的高度。
        - W：图像的宽度。

        返回：
        - x：输出张量，shape为[B, N, out_features]，其中out_features为输出通道数。

        注意：输入张量x的shape中的N要与H * W相等。
        '''
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)

        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        sr_ratio: int = 1,
    ):
        """
        ShiftedBlock module used in a neural network model.

        Args:
            dim: Input dimension of the tensor.
            num_heads: Number of attention heads.
            mlp_ratio: Multiplier for the hidden dimension of the MLP.
            qkv_bias: Whether to include a bias term for qkv linear layers.
            qk_scale: Scale factor for the query and key dimensions.
            drop: Dropout rate for the MLP and attention output.
            attn_drop: Dropout rate for attention weights.
            drop_path: Dropout rate for the residual connections.
            act_layer: Activation function to be used in the MLP.
            norm_layer: Normalization layer used in the model.
            sr_ratio: Spatial reduction ratio.

        """
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            constant_(m.bias, 0)
            constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Forward pass of the ShiftedBlock module.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            H: Height of the input tensor (used for spatial shifting).
            W: Width of the input tensor (used for spatial shifting).

        Returns:
            Output tensor after applying the module.

        """
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class OverlapPatchEmbed(nn.Module):
    """
    Image to Patch Embedding module.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Initializes the OverlapPatchEmbed module.

        Args:
            img_size: Size of the input image (assumed to be square).
            patch_size: Size of each image patch.
            stride: Stride value for overlapping patches.
            in_chans: Number of input image channels.
            embed_dim: Dimension of the embedded patches.

        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H = img_size[0] // patch_size[0]
        self.W = img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """
        Initializes the weights of the module.

        Args:
            m: Module to initialize the weights for.

        """
        if isinstance(m, nn.Linear):
            kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            constant_(m.bias, 0)
            constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, int, int]:
        """
        Forward pass of the OverlapPatchEmbed module.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Embedded patches tensor, height, and width.

        """
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class UNext(nn.Module):

    """这段代码定义了一个名为"UNext"的类，该类继承自nn.Module。代码开始的部分是初始化函数__init__，其中定义了许多类属性和模块的实例化，以及它们的参数。下面是代码的解析：

        在初始化函数__init__中，首先调用了父类的初始化函数super().init()，然后定义了一些类属性和模块的实例化。

        在模块的实例化部分，首先定义了三个卷积层self.encoder1、self.encoder2、self.encoder3，它们分别使用不同的输入和输出通道数进行卷积操作。

        然后定义了三个批归一化层self.ebn1、self.ebn2、self.ebn3，用于对卷积层的输出进行归一化。

        接着定义了两个LayerNorm层self.norm3、self.norm4，用于对特征图的每个通道进行归一化。

        然后定义了两个nn.ModuleList对象self.block1、self.block2，其中每个对象中包含一个shiftedBlock模块。

        随后，定义了两个OverlapPatchEmbed对象self.patch_embed3、self.patch_embed4，用于将输入图像切分成重叠的图像块。

        然后定义了五个卷积层self.decoder1、self.decoder2、self.decoder3、self.decoder4、self.decoder5，用于进行图像解码。

        最后定义了三个BatchNorm2d层self.dbn1、self.dbn2、self.dbn3，用于对解码器的输出进行归一化。

        定义了一个卷积层self.final，用于将解码器的输出映射到指定的类别数量。

        定义了一个Softmax层self.soft，用于对输出进行softmax操作。

        forward方法是类的前向传播函数，接收一个输入张量x，并通过一系列的操作将其映射为最终的输出。具体过程如下：

        首先，对输入进行了一系列的卷积、池化和批归一化操作，得到三个不同尺度的特征图t1、t2、t3。

        接着，将第三个尺度的特征图输入到patch_embed3中，得到out。

        然后，通过一个for循环，将out输入到block1中进行一系列的操作，得到特征图out。

        对out进行归一化，并重新调整其形状，得到out。

        将out输入到patch_embed4中，得到out。

        再次通过一个for循环，将out输入到block2中进行一系列的操作，得到特征图out。

        对out进行归一化，并重新调整其形状，得到out。

        对out进行反卷积操作，并与t4相加，得到特征图out。

        对out进行一系列的操作，得到特征图out。

        对out进行反卷积操作，并与t3相加，得到特征图out。

        对out进行一系列的操作，得到特征图out。

        对out进行反卷积操作，并与t2相加，得到特征图out。

        对out进行反卷积操作，并与t1相加，得到特征图out。

        对out进行反卷积操作，得到最终的输出特征图out。

        将out输入到最后的卷积层self.final中，得到输出。

    代码中包含了许多其他函数和操作，如重塑张量形状，使用ReLU激活函数，插值操作等。整体上，这段代码定义了一个UNext模型，用于进行图像的特征提取和解码。
    """

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)  
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)  
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)
        
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1,padding=1)  
        self.decoder2 =   nn.Conv2d(160, 128, 3, stride=1, padding=1)  
        self.decoder3 =   nn.Conv2d(128, 32, 3, stride=1, padding=1) 
        self.decoder4 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)
        
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,t4)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)
    

class UNext_S(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP w less parameters
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[32, 64, 128, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(3, 8, 3, stride=1, padding=1)  
        self.encoder2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)  
        self.encoder3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(8)
        self.ebn2 = nn.BatchNorm2d(16)
        self.ebn3 = nn.BatchNorm2d(32)
        
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(64)
        self.dnorm4 = norm_layer(32)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(128, 64, 3, stride=1,padding=1)  
        self.decoder2 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)  
        self.decoder3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1) 
        self.decoder4 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(8, 8, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(64)
        self.dbn2 = nn.BatchNorm2d(32)
        self.dbn3 = nn.BatchNorm2d(16)
        self.dbn4 = nn.BatchNorm2d(8)
        
        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,t4)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)

