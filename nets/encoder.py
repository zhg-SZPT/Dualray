import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt



# class ResNet101(nn.Module):
#     def __init__(self, num_classes, **kwargs):
#         super(ResNet101, self).__init__()
#
#         resnet101 = torchvision.models.resnet101(pretrained=True)
#         self.base = nn.Sequential(*list(resnet101.children())[:-2])
#         self.classifier = nn.Linear(2048, num_classes)
#         self.feat_dim = 2048  # feature dimension
#         self.Att = Attention(dim=2048)
#
#     def forward(self, x1, x2):
#         x1 = self.base(x1)
#         x2 = self.base(x2)
#         x = self.Att(x1, x2)
#         x = F.avg_pool2d(x, x.size()[2:])
#         f = x.view(x.size(0), -1)
#         y = self.classifier(f)
#         return y

class Attention(nn.Module):
    def __init__(self,
                 dim=1024,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)  # 防止过拟合

    def forward(self, x1, x2):
        # [batch_size, num_patches + 1, total_embed_dim]
        m = x1.size(2)
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)  # torch.Size([1, 43264, 64])
        B, N, C = x1.shape
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv1 = self.qkv(x1) # torch.Size([1, 43264, 192])
        qkv1 = qkv1.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)    # torch.Size([3, 1, 8, 43264, 8])
        qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)    # torch.Size([3, 1, 8, 43264, 8])
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # make torchscript happy (cannot use tensor as tuple)
        q1 = qkv1[0, :, :, :, :]
        k1 = qkv1[1, :, :, :, :]
        v1 = qkv1[2, :, :, :, :]
        q2 = qkv2[0, :, :, :, :]
        k2 = qkv2[1, :, :, :, :]
        v2 = qkv2[2, :, :, :, :]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)  # 进行点乘后的数值很大，导致通过softmax后梯度变的很小
        attn1 = self.attn_drop(attn1)
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale  # attention公式
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)

        # picture_hot
        # attn_hot_train = attn1.view(8, 64, 64)[0].cpu().detach().numpy()
        # np.save('./save_attn_hot_train', attn_hot_train)
        # attn_hot_moudle = attn2.view(8, 64, 64)[0].cpu().detach().numpy()
        # np.save('./save_attn_hot_moudle', attn_hot_moudle)
        # sns.heatmap(attn_hot_train, fmt=".1f", cmap="RdBu_r")
        # plt.show()
        # sns.heatmap(attn_hot_moudle, fmt=".1f", cmap="RdBu_r")
        # plt.show()


        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        x1 = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)
        x = (x+x1)/2
        x = self.proj(x)  # linner
        x = self.proj_drop(x)
        x = x.transpose(1, 2)
        return x.reshape(x.size()[0], x.size()[1], m, -1)


