'''
@Author: Qin Yang
@Date: 2020-03-12 12:16:41
@Email: qinyangforever@foxmail.com
@LastEditors: Qin Yang
@LastEditTime: 2020-03-12 21:53:12
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['Generator', 'Discriminator']


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.relu = nn.LeakyReLU(0.3)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, x):
        rx = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return (x+rx)


class UpsampleBlock(nn.Module):
    def __init__(self, num_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor,
                             mode='linear', align_corners=True)


class PixelShuffle1d(nn.Module):
    def __init__(self, num_channels, scale_factor=2):
        super(PixelShuffle1d, self).__init__()

        self.num_channels = num_channels
        self.conv = nn.Conv1d(num_channels, num_channels *
                              scale_factor, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        batch_size = x.size(0)
        num_channels = self.num_channels

        x = x.view(batch_size, num_channels, 2, - 1)
        x = torch.transpose(x, 2, 3).contiguous()
        return x.view(batch_size, num_channels, -1)


class Generator(nn.Module):
    def __init__(self, c_dim=12, num_channels=64, scale_factor=4., ScaleBlockType=UpsampleBlock):
        super(Generator, self).__init__()

        upsampling_num_blocks = int(math.log2(scale_factor))

        self.rand_block = nn.Sequential(
            nn.Conv1d(c_dim, num_channels, kernel_size=9, padding=4),
            nn.LeakyReLU(0.3)
        )

        blocks = [ResidualBlock(num_channels)]
        for _ in range(upsampling_num_blocks):
            blocks.append(ScaleBlockType(num_channels))
            blocks.append(ResidualBlock(num_channels))
        self.blocks = nn.Sequential(*blocks)

        self.final_block = nn.Conv1d(
            num_channels, c_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.rand_block(x)
        x = self.blocks(x)
        x = self.final_block(x)
        return F.tanh(x) / 2


class Discriminator(nn.Module):
    def __init__(self, c_dim=12, num_layers=6, num_channels=64):
        super(Discriminator, self).__init__()

        self.pri_block = nn.Sequential(
            nn.Conv1d(c_dim, num_channels, kernel_size=9, padding=4),
            nn.LeakyReLU(0.3)
        )

        blocks = [ResidualBlock(num_channels)]
        for _ in range(num_layers):
            blocks.append(nn.AvgPool1d(kernel_size=3, stride=2, padding=1))
            blocks.append(ResidualBlock(num_channels))
        self.blocks = nn.Sequential(*blocks)

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels, 1, bias=True)

    def forward(self, x):
        x = self.   i_block(x)
        x = self.blocks(x)
        x = self.global_pooling(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
